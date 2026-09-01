import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible instance normalization, input side only.

    Attacks the raw-scale defect of this pipeline: prices enter the network at their
    absolute level (tens of thousands of USD) and the evaluation period trades above
    anything seen in training, so the network is asked to extrapolate outside the range
    its weights were fitted on. Removing per-window location and scale makes every
    window look alike regardless of what BTC happened to cost that year.

    Only the normalization half of RevIN is used. The published method also
    de-normalizes the model output, but this backbone predicts a relative return that
    ``mode='ret'`` reconstructs around the last observed close, so the price level never
    passes through the network and there is nothing to restore.

    ``alpha`` in [0, 1] sets how much of the per-window standard deviation is removed
    (1.0 = the usual full normalization, 0.0 = subtract the mean only). Alpha-RevIN
    reports that removing all of it imposes an overly strong prior on some series,
    which is why one arm of this round keeps a fraction of the original scale.

    Input and output are ``(B, F, T)`` — statistics are taken over ``T`` per channel,
    per sample. Nothing is shared across the batch, so no information crosses between
    windows and none crosses a split boundary.
    """

    def __init__(self, num_features, eps=1e-5, alpha=1.0, affine=True):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'alpha must be in [0, 1], got {alpha}')
        self.num_features = num_features
        self.eps = eps
        self.alpha = alpha
        self.affine = affine
        if affine:
            # Learned per-channel rescaling: the network can put back whatever amount
            # of scale actually helps instead of being forced to treat every channel
            # as standardised.
            self.weight = nn.Parameter(torch.ones(num_features, 1))
            self.bias = nn.Parameter(torch.zeros(num_features, 1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # Biased std: a window is the whole population here, not a sample of one.
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        # clamp_min already keeps this away from zero, so no second epsilon in the
        # denominator — at alpha=0 the divisor must be exactly 1, not 1 + eps.
        scale = std.clamp_min(self.eps) ** self.alpha
        out = (x - mean) / scale
        if self.affine:
            out = out * self.weight + self.bias
        return out

    def extra_repr(self):
        return f'num_features={self.num_features}, alpha={self.alpha}, affine={self.affine}'
