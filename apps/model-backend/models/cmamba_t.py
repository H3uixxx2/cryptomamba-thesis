import torch.nn as nn

from models.blueprint_blocks import (BiScanWrapper, RecencyQueryPool,
                                     initialize_mamba_timescales)
from models.cmamba import CMBlock
from models.revin import RevIN


class CMambaT(nn.Module):
    """Time-axis CryptoMamba: the SSM scans the DAYS, not the features.

    Upstream CMamba is "inverted": the Mamba scan runs across the 6 feature tokens
    while the time window is the channel dim, mixed only by Linear layers. That (a)
    puts Mamba's selective forgetting on the wrong axis and (b) hard-couples window
    length to model width (window == d_model), capping context at 14 days.

    CMambaT decouples them:
      input  (B, F, T)  [repo tensor layout]
      embed  per-day Linear F -> d_model                  -> (B, T, d_model)
      body   n_blocks x CMBlock(d_model), causal scan over the T day-tokens
      head   last day's representation -> LayerNorm -> Linear -> 1
    Pair with mode='ret' so the output is a relative return around the last close.
    """

    def __init__(self, num_features=5, window_size=14, d_model=32, n_blocks=4,
                 d_state=16, d_conv=4, expand=2, mlp_ratio=2, drop=0.1,
                 revin=False, revin_alpha=1.0, revin_affine=True,
                 distributional=False, head_outputs=None,
                 bi_scan=False, recency_pool=False, timescale_init=False,
                 eca_kernel=5, tau_min=2.0, tau_max=60.0, **kwargs):
        super().__init__()
        self.window_size = window_size
        # Off by default: every previously reported run must keep behaving identically.
        # When on, pair it with feature_flags.window_norm=False in the training config —
        # the two do the same job and stacking them normalises twice.
        self.revin = RevIN(num_features, alpha=revin_alpha, affine=revin_affine) if revin else None
        self.embed = nn.Linear(num_features, d_model)
        self.blocks = nn.ModuleList([
            CMBlock(hidden_dim=d_model, d_state=d_state, d_conv=d_conv,
                    expand=expand, mlp_ratio=mlp_ratio, drop=drop, **kwargs)
            for _ in range(n_blocks)
        ])
        # Blueprint §2 pieces, all default-off and all identity at init, so an arm that
        # switches one on starts as a numerically exact copy of the control.
        if timescale_init:
            for blk in self.blocks:
                initialize_mamba_timescales(blk.op, tau_min=tau_min, tau_max=tau_max)
        if bi_scan:
            self.blocks = nn.ModuleList([
                BiScanWrapper(blk, d_model, eca_kernel=eca_kernel)
                for blk in self.blocks])
        self.pool = RecencyQueryPool(d_model) if recency_pool else None
        self.norm = nn.LayerNorm(d_model)
        # Two outputs in distributional mode: the return, and log-sigma relative to
        # the last close. Width 1 keeps every existing checkpoint loadable.
        # head_outputs overrides it for heads that need a third channel (the centred
        # scale mixture needs a tail logit).
        if head_outputs is None:
            head_outputs = 2 if distributional else 1
        self.head = nn.Linear(d_model, head_outputs)

    def forward(self, x):
        if self.revin is not None:
            x = self.revin(x)                # (B, F, T), per-window per-channel
        h = self.embed(x.permute(0, 2, 1))   # (B, F, T) -> (B, T, d_model)
        for blk in self.blocks:
            h = blk(h)
        # Causal readout: the most recent day, or a recency-weighted pool over the window.
        h = self.norm(h[:, -1] if self.pool is None else self.pool(h))
        return self.head(h)
