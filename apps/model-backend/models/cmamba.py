import math
from functools import partial
from typing import Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import torch.utils.checkpoint as checkpoint
try:  # native CUDA kernels (Colab/Linux GPU). On CPU-only hosts (local macOS) fall back
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:  # pragma: no cover - kernels absent
    causal_conv1d_fn, causal_conv1d_update = None, None
try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
except ImportError:  # pragma: no cover - kernels absent
    selective_state_update, selective_scan_fn, mamba_inner_fn = None, None, None


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                       delta_softplus=False, return_last_state=False):
    """Pure-PyTorch reference of mamba_ssm's selective_scan (same math, differentiable).

    Used only when the CUDA kernels are unavailable; the fused-kernel path is untouched.
    u: (b, d, l); delta: (b, d, l); A: (d, n); B, C: (b, n, l); D: (d,); z: (b, d, l)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    B = B.float()
    C = C.float()
    x = A.new_zeros((u.shape[0], A.shape[0], A.shape[1]))
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
    ys = []
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        ys.append(torch.einsum('bdn,bn->bd', x, C[:, :, i]))
        if i == u.shape[2] - 1:
            last_state = x
    y = torch.stack(ys, dim=2)  # (b, d, l)
    if D is not None:
        y = y + u * rearrange(D.float(), 'd -> d 1')
    if z is not None:
        y = y * F.silu(z.float())
    out = y.to(dtype=dtype_in)
    return (out, last_state) if return_last_state else out


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape
        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        # kernels are CUDA-only: route by the tensor's device, not by import success,
        # so a CPU tensor on a CUDA box still takes the pure-PyTorch fallback
        if self.use_fast_path and causal_conv1d_fn is not None and hidden_states.is_cuda:
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if causal_conv1d_fn is None or not x.is_cuda:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            scan_fn = selective_scan_fn if (selective_scan_fn is not None and x.is_cuda) else selective_scan_ref
            y = scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=False,
            )
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out





class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class CMBlock(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        d_state: int = 16,
        dt_rank: Any = "auto",
        d_conv=4,
        expand=2,
        use_checkpoint: bool = False,
        mlp_ratio=2,
        act_layer=nn.ReLU,
        drop: float = 0.0,
        **kwargs,
    ): 
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = norm_layer(hidden_dim)

        self.op = Mamba(d_model=hidden_dim,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        dt_rank=dt_rank,
                        **kwargs
                        )
        
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=False)
            # _forward references drop_path but it was never defined -> mlp_ratio>0 crashed.
            self.drop_path = nn.Identity()

    def _forward(self, x):
        h = self.op(self.norm(x))
        # h = self.op(x)
        h += x
        if self.mlp_branch:
            h = h + self.drop_path(self.mlp(self.norm2(h)))
        return h

    def forward(self, x):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, (x))
        else:
            return self._forward(x)


class CMamba(nn.Module):

    def __init__(
        self,
        num_features=5,
        hidden_dims=[14, 1],
        norm_layer=nn.LayerNorm,
        d_conv=4,
        layer_density=1,
        expand=2, 
        mlp_ratio=0, 
        drop=0.0, 
        num_classes=None,
        d_states=16,
        use_checkpoint=False,
        cls=False,
        **kwargs
    ):
        super().__init__()

        self.hidden_dims = hidden_dims
        self.expand = expand
        self.mlp_ratio = mlp_ratio
        self.drop = drop
        self.num_features = num_features
        self.d_conv = d_conv
        self.layer_density = None
        self.num_classes = num_classes
        self.norm_layer = norm_layer
        self.d_states = None
        self.use_checkpoint = use_checkpoint
        self._set_d_states(d_states)
        self._create_layer_density(layer_density) 
        self.args = kwargs
        self.act = nn.ReLU
        self.cls = cls

        self.post_process = nn.Sequential(
            Permute(0, 2, 1),
            nn.Linear(num_features, 1),
        )
        self.tanh = nn.Tanh()

        d = len(hidden_dims)
        self.blocks = nn.ModuleList(
            self._get_block(hidden_dims[i], hidden_dims[i + 1], self.layer_density[i], self.d_states[i])
            for i in range(d - 1)
        )

        # self.norm = norm_layer((num_features, hidden_dims[0]))
        self.activation = self.act()

    
    def _set_d_states(self, d_states):
        n = len(self.hidden_dims)
        # if d_states == None:
        #     self.d_states = ['auto' for _ in range(n)]
        if isinstance(d_states, list):
            self.d_states = d_states
        else:
            self.d_states = [d_states for _ in range(n)]


    def _create_layer_density(self, layer_density):
        n = len(self.hidden_dims)
        if not isinstance(layer_density, list):
            self.layer_density = [layer_density for _ in range(n)]
        else:
            self.layer_density = layer_density

    def _get_block(self, hidden_dim, hidden_dim_next, n, d_state):
        # print(f'ds - {hidden_dim} - {n}')
        modules = [CMBlock(hidden_dim=hidden_dim,
                           norm_layer=self.norm_layer,
                           d_state=d_state,
                           d_conv=self.d_conv,
                           expand=self.expand,
                           use_checkpoint=self.use_checkpoint,
                           mlp_ratio=self.mlp_ratio,
                           act_layer=self.act,
                           drop=self.drop,
                           **self.args
                           ) 
                           for _ in range(n)]
        modules.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim_next))
        # modules.append(self.norm_layer(hidden_dim_next))
        return nn.Sequential(*modules)

    
    def forward(self, x):
        # x = self.norm(x)
        for layer in self.blocks:
            x = layer(x)

        x = self.post_process(x)
        if self.cls:
            x = self.tanh(x)
        return x