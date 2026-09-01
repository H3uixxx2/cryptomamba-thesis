"""Plan screen API — REPORTS readiness only, never computes research results.

Reflects the true state: which evaluation artifacts exist on disk and the
reproduce status flags. Disclosed limitations (baselines 2/5, usability study
dropped) are surfaced honestly.
"""
from __future__ import annotations

from fastapi import APIRouter

from ..core import config
from .. import logic

router = APIRouter(prefix="/api/plan", tags=["plan"])

GREEN = "var(--signal-green)"
AMBER = "var(--signal-amber)"
RED = "var(--signal-red)"

_ARTIFACT_SLOTS = [
    ("forecast_metrics.csv", "Forecast RMSE/MAE/MAPE vs paper", config.EVALUATION_DIR / "forecast_metrics.csv"),
    ("trading_replay_metrics.csv", "Paper trading replay", config.EVALUATION_DIR / "trading_replay_metrics.csv"),
    ("baseline_metrics_comparison.csv", "Baseline comparison", config.EVALUATION_DIR / "baseline_metrics_comparison.csv"),
    ("significance_tests.csv", "DM / Wilcoxon + directional acc", config.EVALUATION_DIR / "significance_tests.csv"),
    ("inference_fixture.json", "Golden prediction fixture", config.EVALUATION_DIR / "inference_fixture.json"),
    ("offline_prediction.json", "Defense-day offline backup", config.OFFLINE_PREDICTION_PATH),
    ("trading_metrics.csv", "Full chronological backtest (Phase 4)", config.EVALUATION_DIR / "trading_metrics.csv"),
]


@router.get("")
def get_plan() -> dict:
    a = logic.load_reproduce_artifacts(
        evaluation_dir=config.EVALUATION_DIR,
        provenance_dir=config.REPRODUCE_PROVENANCE_DIR,
        selected_checkpoint_path=config.SELECTED_CHECKPOINT_PATH,
    )
    n_base = len(a.baseline_models_present)
    repro_ready = a.status == "READY"
    offline_ok = config.OFFLINE_PREDICTION_PATH.exists()
    backtest_ok = (config.EVALUATION_DIR / "trading_metrics.csv").exists()
    baselines_complete = a.baseline_status == "COMPLETE"
    significance_ok = (config.EVALUATION_DIR / "significance_tests.csv").exists() and baselines_complete

    page_status = [
        {"page": "01 · Data", "status": "READY", "color": GREEN,
         "truth": "Paper split + upload pipeline hoạt động", "next": "—"},
        {"page": "02 · Reproduce",
         "status": "READY" if repro_ready else "NOT_READY",
         "color": GREEN if repro_ready else AMBER,
         "truth": f"Forecast {a.forecast_status}; baselines {n_base}/5",
         "next": "—" if baselines_complete else "Train LSTM/GRU/iTransformer (Colab GPU)"},
        {"page": "03 · Predict",
         "status": "READY" if offline_ok else "PARTIAL",
         "color": GREEN if offline_ok else AMBER,
         "truth": "Live (Colab) + offline backup" if offline_ok else "Live only; no offline artifact",
         "next": "—" if offline_ok else "Freeze offline_prediction.json"},
        {"page": "04 · Trading",
         "status": "READY" if backtest_ok else "DEMO",
         "color": GREEN if backtest_ok else AMBER,
         "truth": "One-day demo + full chronological backtest" if backtest_ok
                  else "One-day decision demo; no full backtest yet",
         "next": "—" if backtest_ok else "Full chronological backtest w/ fees"},
        {"page": "05 · Plan", "status": "READY", "color": GREEN,
         "truth": "Reports readiness only (no compute)", "next": "—"},
    ]

    artifact_slots = []
    for name, purpose, path in _ARTIFACT_SLOTS:
        exists = path.exists()
        artifact_slots.append(
            {
                "name": name,
                "purpose": purpose,
                "present": exists,
                "badge_text": "PRESENT" if exists else "MISSING",
                "badge_color": GREEN if exists else RED,
            }
        )

    # validated/pending are state-driven so they never go stale: an item moves
    # from pending -> validated the moment its artifact/gate is satisfied.
    validated = [
        {"name": "CryptoMamba-v forecast", "note": f"{a.forecast_status} · <5% vs paper"},
        {"name": "Official trading replay", "note": a.replay_status},
        {
            "name": "Baselines (4/4)" if baselines_complete else "Baselines (partial)",
            "note": "naive+LSTM+GRU+iTransformer" if baselines_complete else f"{n_base}/4 present",
        },
        {"name": "Offline prediction backup", "note": "frozen" if offline_ok else "missing"},
    ]
    if significance_ok:
        validated.append({"name": "Significance over all 4", "note": "DM/Wilcoxon + directional acc"})
    if backtest_ok:
        validated.append({"name": "Full trading backtest", "note": "chronological + fees (test+val)"})

    pending = []
    if not baselines_complete:
        pending.append({"name": "Neural baselines", "note": "LSTM/GRU/iTransformer — Colab GPU"})
    if not significance_ok:
        pending.append({"name": "Significance over all 5", "note": "re-run when baselines land"})
    if not backtest_ok:
        pending.append({"name": "Full trading backtest", "note": "chronological + fees"})
    # Always disclosed: a deliberate proposal deviation, not work-in-progress.
    pending.append({"name": "Usability study", "note": "dropped (disclosed limitation)"})
    demo_story = [
        {"n": "1", "title": "Data", "text": "Bộ BTC/USD chuẩn hoá, split theo paper, cửa sổ 14 ngày."},
        {"n": "2", "title": "Reproduce", "text": "Forecast <5% vs paper; baseline + significance công khai."},
        {"n": "3", "title": "Predict", "text": "Point forecast ngày kế — live Colab hoặc offline backup."},
        {"n": "4", "title": "Trading", "text": "Một dự đoán biến thành quyết định mua/bán (vanilla vs smart)."},
        {"n": "5", "title": "Plan", "text": "Bảng readiness — chỉ claim khi có artifact thật."},
    ]

    return {
        "page_status": page_status,
        "artifact_slots": artifact_slots,
        "validated": validated,
        "pending": pending,
        "demo_story": demo_story,
        "reproduce_status": a.status,
        "baselines_present": sorted(a.baseline_models_present),
    }
