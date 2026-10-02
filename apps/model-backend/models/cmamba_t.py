import torch.nn as nn

from models.cmamba import CMBlock


class CMambaT(nn.Module):
    """Time-axis CryptoMamba: the SSM scans the days, not the features, so window length is independent of model width.

      input  (B, F, T)  [repo tensor layout]
      embed  per-day Linear F -> d_model                  -> (B, T, d_model)
      body   n_blocks x CMBlock(d_model), causal scan over the T day-tokens
      head   last day's representation -> LayerNorm -> Linear -> 1

    Pair with mode='ret' so the output is a relative return around the last close.
    """

    def __init__(self, num_features=5, window_size=14, d_model=32, n_blocks=4,
                 d_state=16, d_conv=4, expand=2, mlp_ratio=2, drop=0.1, **kwargs):
        super().__init__()
        self.window_size = window_size
        self.embed = nn.Linear(num_features, d_model)
        self.blocks = nn.ModuleList([
            CMBlock(hidden_dim=d_model, d_state=d_state, d_conv=d_conv,
                    expand=expand, mlp_ratio=mlp_ratio, drop=drop, **kwargs)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.embed(x.permute(0, 2, 1))   # (B, F, T) -> (B, T, d_model)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h[:, -1])              # causal readout: the most recent day
        return self.head(h)
