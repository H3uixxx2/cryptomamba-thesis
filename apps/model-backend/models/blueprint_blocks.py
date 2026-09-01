"""Architecture pieces from the 2026-07-29 blueprint, §2 (RegimeBiScan-Lite).

Adapted rather than copied: the reference module in
`.ai/CryptoMamba_SSM_Research_Blueprint_2026-07-29/` assumes a bare scan of shape
[B, T, d] -> [B, T, d], but this repo's `CMBlock` already carries its own norm and
residual. Wrapping it with the reference block verbatim would normalise and add the
residual twice.

Everything here is **identity at initialisation**, which the reference is not: the
reference sets the residual scale to 0.1, so an ablation would start from a model that is
already different. Here the mixing weight starts at exactly 0, so the bi-scan arm and the
control arm begin numerically identical and diverge only as the weight is learned. That
makes the comparison a real ablation instead of two different models.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelGate(nn.Module):
    """ECA-style channel recalibration: k weights, no squeeze/excitation MLP.

    Zero-initialised conv means sigmoid(0) = 1/2 and the gain is exactly 2 * 1/2 = 1, so
    this starts as the identity. Reference: ECA-Net (arXiv 1910.03151), whose published
    gains are in vision — hence it ships inside the ablated arm, not the control.
    """

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError('kernel_size must be a positive odd integer')
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError('expected [batch, time, channel]')
        summary = x.mean(dim=1).unsqueeze(1)                  # [B, 1, d]
        gain = 2.0 * torch.sigmoid(self.conv(summary)).squeeze(1)
        return x * gain.unsqueeze(1)


class BiScanWrapper(nn.Module):
    """Run one causal block in both time directions with the SAME weights, then blend.

    The reverse pass costs compute, not parameters. Non-leaking only because the window is
    a fixed slice of completed candles [t-59, t]: within that slice the reverse pass does
    mix later days into earlier positions, which is exactly the point, and which is also
    why this must never be handed a centred or future-padded window.

    Cost per wrapped block: 2*d + 6 parameters (gate gain, gate bias, blend weight, ECA).
    """

    def __init__(self, block: nn.Module, d_model: int, eca_kernel: int = 5):
        super().__init__()
        self.block = block
        self.gate_gain = nn.Parameter(torch.zeros(d_model))
        self.gate_bias = nn.Parameter(torch.zeros(d_model))
        # Zero, not 0.1: the arm must start as an exact copy of the control.
        self.blend = nn.Parameter(torch.zeros(()))
        self.channel_gate = ChannelGate(eca_kernel)

    def forward(self, x):
        forward = self.block(x)
        reverse = torch.flip(self.block(torch.flip(x, dims=(1,))), dims=(1,))
        gate = torch.sigmoid(x * self.gate_gain + self.gate_bias)
        # blend = 0 collapses to `forward`; the gate only decides how much of the reverse
        # pass replaces it, per channel and per day.
        return self.channel_gate(forward + self.blend * gate * (reverse - forward))


class RecencyQueryPool(nn.Module):
    """Read out with one learned query and an age penalty instead of the last day alone.

    d + 2 parameters. The mixing logit starts strongly negative so the pool begins as
    "the last day", matching the control's readout, and only learns to reach further back
    if that helps. Lets the model retrieve a range shock from mid-window without a dense
    T*d temporal projection.
    """

    def __init__(self, d_model: int, initial_mix: float = 0.02):
        super().__init__()
        self.query = nn.Parameter(torch.empty(d_model))
        self.log_decay = nn.Parameter(torch.tensor(-2.0))
        self.mix_logit = nn.Parameter(torch.tensor(math.log(initial_mix
                                                            / (1.0 - initial_mix))))
        nn.init.normal_(self.query, std=d_model ** -0.5)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError('expected [batch, time, channel]')
        age = torch.arange(x.shape[1] - 1, -1, -1, device=x.device, dtype=x.dtype)
        score = (torch.einsum('btd,d->bt', x, self.query) / math.sqrt(x.shape[-1])
                 - F.softplus(self.log_decay) * age.unsqueeze(0))
        context = torch.einsum('bt,btd->bd', torch.softmax(score, dim=1), x)
        mix = torch.sigmoid(self.mix_logit)
        return (1.0 - mix) * x[:, -1] + mix * context


@torch.no_grad()
def initialize_mamba_timescales(mamba: nn.Module, tau_min: float = 2.0,
                                tau_max: float = 60.0) -> None:
    """Spread the initial state e-folding times log-uniformly over [tau_min, tau_max] days.

    A = -exp(A_log) and the discrete decay is exp(delta * A), so setting
    A_n = -1/(delta_0 * tau_n) makes state n start with a memory of about tau_n days at
    the typical initial step size. Reparameterises existing weights: zero new parameters.

    Raises if the module does not expose the attribute names this depends on, so a silent
    no-op is impossible.
    """
    for name in ('A_log', 'dt_proj', 'd_state', 'd_inner'):
        if not hasattr(mamba, name):
            raise TypeError(f'scan does not expose required Mamba attribute {name!r}')
    if not 0.0 < tau_min < tau_max:
        raise ValueError('require 0 < tau_min < tau_max')
    dt0 = F.softplus(mamba.dt_proj.bias.detach().float()).median().clamp_min(1e-6)
    tau = torch.exp(torch.linspace(math.log(tau_min), math.log(tau_max),
                                   int(mamba.d_state), dtype=torch.float32,
                                   device=mamba.A_log.device))
    target = torch.log(1.0 / (dt0 * tau)).unsqueeze(0).expand(int(mamba.d_inner), -1)
    mamba.A_log.copy_(target.to(dtype=mamba.A_log.dtype))
