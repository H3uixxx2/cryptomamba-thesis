"""Training-run evidence: per-seed results, seed-family comparisons and the ablation, read from the sealed ``runs/`` tables.
A missing or unlisted file yields status=NOT_READY. Families: CryptoMamba-v (``cmamba_v_reproduced``) and CryptoMamba-T (``s5_full``), seeds 23-25;
``ABLATION_MODELS`` names the five intermediate models.
"""
from __future__ import annotations

import re
from typing import Any

from ..core import config
from ..loaders.final_evidence import EvidenceIntegrityError, FinalEvidence, records

SEEDS = (23, 24, 25)

# Internal run id -> (name shown in the console, what differs from CM-v).
ABLATION_MODELS: tuple[tuple[str, str, str], ...] = (
    ("CM-v", "CryptoMamba-v", "As published: feature-axis scan, raw inputs, price output, 14 days"),
    ("preproc", "CM-v, return output", "Output changed to a relative return"),
    ("B1b", "CM-v, normalised input", "Return output and CryptoMamba-T's input normalisation"),
    ("arch", "Time axis, 14 days, lr 0.01", "Scan moved to the time axis, learning rate 0.01"),
    ("B2", "Time axis, 14 days, lr 0.001", "Time axis, learning rate 0.001"),
    ("B3", "CM-v, change output", "Output changed to a change from today's close"),
    ("CM-T", "CryptoMamba-T", "Time axis, 60 days, lr 0.001"),
)
FIRST_VERSION = ("B1", "CM-v, normalised input, lr 0.001", "First version of comparison 3 (one seed)")

_RUN_KEY = re.compile(r"^(?P<model>.+) s(?P<seed>\d+)\|(?P<split>val|test)$")
_METRIC_COLUMNS = (
    "member", "model_id", "split", "samples", "RMSE", "MAE", "MAPE_pct",
    "directional_accuracy_pct", "predicted_up_pct", "row_kind",
)


def _ablation_models(per_run: dict[str, Any]) -> list[dict[str, Any]]:
    runs: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    naive: dict[str, dict[str, Any]] = {}
    for key, value in per_run.items():
        if key.startswith("naive|"):
            naive[key.split("|", 1)[1]] = value
            continue
        match = _RUN_KEY.match(key)
        if match is None:
            continue  # the "<model> mean|split" and "<model> range|split" summary rows
        runs.setdefault((match["model"], match["split"]), {})[int(match["seed"])] = value

    models = []
    for run_id, name, note in (*ABLATION_MODELS, FIRST_VERSION):
        by_split = {}
        for split in ("val", "test"):
            seeds = runs.get((run_id, split), {})
            if not seeds:
                continue
            rmse = [seeds[s]["rmse"] for s in sorted(seeds)]
            by_split[split] = {
                "runs": [{"seed": s, **seeds[s]} for s in sorted(seeds)],
                "rmse_mean": sum(rmse) / len(rmse),
                "rmse_min": min(rmse),
                "rmse_max": max(rmse),
            }
        if by_split:
            models.append({"id": run_id, "name": name, "note": note, "splits": by_split})
    return models


def _sign_direction(metrics, direction) -> list[dict[str, Any]]:
    """Directional accuracy by sign agreement on every date.
    ``controlled_metrics.csv`` excludes dates whose forecast is within a relative 1e-6 of today's close (two validation cells move 0.15-0.17 points);
    the interval file covers every date, so its point estimates replace that column.
    """
    primary = direction[(direction["kind"] == "model") & (direction["block_length"] == 7)]
    by_run = {
        (str(r.member), r.model_id, r.split): float(r.estimate)
        for r in primary.itertuples()
        if str(r.member) != "family"
    }
    out = metrics.copy()
    for index, row in out.iterrows():
        if row["model_id"] == "naive_persistence":
            continue
        split, model = row["split"], row["model_id"]
        if row["row_kind"] == "member":
            value = by_run.get((str(row["member"]), model, split))
        else:
            seeds = [by_run.get((str(s), model, split)) for s in SEEDS]
            if any(v is None for v in seeds):
                value = None
            else:
                value = {
                    "family_mean": sum(seeds) / len(seeds),
                    "family_min": min(seeds),
                    "family_max": max(seeds),
                }[row["row_kind"]]
        if value is not None:
            out.at[index, "directional_accuracy_pct"] = value
    return records(out)


def build_runs() -> dict[str, Any]:
    try:
        evidence = FinalEvidence(config.FINAL_EVIDENCE_DIR)
        metrics = evidence.read_csv("runs/controlled_metrics.csv")
        comparisons = evidence.read_csv("runs/family_comparisons.csv")
        replay = evidence.read_csv("runs/paper_replay.csv")
        direction = evidence.read_csv("runs/direction_intervals.csv")
        trading = evidence.read_csv("runs/trading_all.csv")
        ablation = evidence.read_json("runs/ablation_results.json")
        ablation_trading = evidence.read_csv("runs/ablation_trading.csv")
        models = _ablation_models(ablation["per_run"])
        ablation_comparisons = ablation["comparisons"]
        naive = {
            split: ablation["per_run"][f"naive|{split}"] for split in ("val", "test")
        }
    except (EvidenceIntegrityError, KeyError, TypeError, ValueError) as exc:
        return {"status": "NOT_READY", "message": str(exc)}

    return {
        "status": "READY",
        "final_evidence_sha256": evidence.package_sha256,
        "seeds": list(SEEDS),
        "metrics": _sign_direction(metrics[[c for c in _METRIC_COLUMNS if c in metrics.columns]], direction),
        "comparisons": records(comparisons),
        "paper_replay": records(replay),
        "trading": records(trading),
        "ablation": {
            "models": models,
            "naive": naive,
            "comparisons": ablation_comparisons,
            "trading": records(ablation_trading),
            "pt": ablation["pt"],
        },
    }
