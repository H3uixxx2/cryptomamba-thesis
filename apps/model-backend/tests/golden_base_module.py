"""Capture BaseModule behaviour on the thesis path (mode default + ret, loss rmse).

Prints a JSON snapshot of forward / training_step / validation_step / test_step /
denormalize / configure_optimizers against fixed seeded tensors. Run it before and after
any edit to base_module.py and diff the two outputs: they must be byte-identical.

    cd apps/model-backend && python tests/golden_base_module.py > /tmp/before.json
    # ...edit...
    python tests/golden_base_module.py > /tmp/after.json && diff /tmp/before.json /tmp/after.json

It stands in a deterministic Linear for CMamba/CMambaT so it runs without mamba_ssm,
i.e. on a plain CPU machine. It checks the BaseModule contract, not the SSM itself.
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import torch, torch.nn as nn
from pl_modules.base_module import BaseModule


class Dummy(nn.Module):
    """Deterministic stand-in for CMamba/CMambaT (mamba_ssm is unavailable here)."""
    def __init__(self, feats, window, out=1):
        super().__init__()
        self.lin = nn.Linear(feats * window, out)
    def forward(self, x):
        return self.lin(x.reshape(x.shape[0], -1))


def make(mode, feats=6, window=14, **kw):
    torch.manual_seed(23)
    m = BaseModule(mode=mode, loss="rmse", window_size=window, **kw)
    m.model = Dummy(feats, window)
    return m


def batch(feats=6, window=14, n=8):
    torch.manual_seed(101)
    return {
        "features": torch.rand(n, feats, window) * 2 - 1,
        "Close": torch.rand(n) * 1000 + 20000,
        "Close_old": torch.rand(n) * 1000 + 20000,
    }


def t(v):
    if torch.is_tensor(v):
        return [round(float(x), 10) for x in v.reshape(-1).tolist()]
    return v


out = {}
for mode, feats, window in (("default", 6, 14), ("ret", 5, 60)):
    m = make(mode, feats, window)
    b = batch(feats, window)
    m.eval()
    with torch.no_grad():
        out[f"{mode}.forward"] = t(m(b["features"], b["Close_old"]))
        y, yo = m.denormalize(b["Close"], b["Close_old"])
        out[f"{mode}.denormalize_y"] = t(y)
    m.train()
    torch.manual_seed(7)
    loss = m.training_step(b, 0)
    out[f"{mode}.training_loss"] = t(loss if torch.is_tensor(loss) else loss["loss"])
    m.eval()
    with torch.no_grad():
        m.validation_step(b, 0)
        m.test_step(b, 0)
    out[f"{mode}.val_test_ran"] = True
    opt = m.configure_optimizers()
    out[f"{mode}.optimizer"] = type(
        opt[0][0] if isinstance(opt, tuple) else (opt["optimizer"] if isinstance(opt, dict) else opt)
    ).__name__
    m.set_normalization_coeffs({"Close": {"min": 1000.0, "max": 70000.0}})
    y2, yo2 = m.denormalize(b["Close"], b["Close_old"])
    out[f"{mode}.denorm_after_coeffs"] = t(y2)
    out[f"{mode}.n_params"] = sum(p.numel() for p in m.parameters())

print(json.dumps(out, indent=1, sort_keys=True))
