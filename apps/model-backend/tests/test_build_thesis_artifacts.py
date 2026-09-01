from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd

from scripts.build_thesis_artifacts import build_all


CORE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FILES = {
    "paper_reported_metrics.csv",
    "controlled_predictions.csv",
    "controlled_forecast_metrics.csv",
    "paired_significance_tests.csv",
    "paper_replay_metrics.csv",
    "corrected_trading_metrics.csv",
    "corrected_trading_equity.csv",
    "corrected_trading_metadata.json",
    "s5_full_summary.json",
    "checkpoint_provenance.json",
    "artifact_manifest.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_outputs_have_the_final_schema(tmp_path: Path) -> None:
    paths = build_all(core_root=CORE_ROOT, output_dir=tmp_path)

    assert set(paths) == EXPECTED_FILES
    assert {path.name for path in tmp_path.iterdir()} == EXPECTED_FILES
    predictions = pd.read_csv(paths["controlled_predictions.csv"])
    metrics = pd.read_csv(paths["controlled_forecast_metrics.csv"])
    tests = pd.read_csv(paths["paired_significance_tests.csv"])
    trading = pd.read_csv(paths["corrected_trading_metrics.csv"])
    assert len(predictions) == 1_824
    assert len(metrics) == 6
    assert len(tests) == 6
    assert len(trading) == 72
    assert trading["reconciliation_error"].le(1e-9).all()


def test_paper_rows_are_separate_aggregate_references(tmp_path: Path) -> None:
    paths = build_all(core_root=CORE_ROOT, output_dir=tmp_path)

    paper = pd.read_csv(paths["paper_reported_metrics.csv"])
    tests = pd.read_csv(paths["paired_significance_tests.csv"])
    assert paper["source_label"].eq("Paper-reported").all()
    assert paper["model_id"].tolist() == [
        "lstm_v",
        "bilstm_v",
        "gru_v",
        "itransformer_v",
        "s_mamba_v",
        "cmamba_v",
    ]
    assert not set(paper["model_id"]) & set(tests["model_a"])
    assert not set(paper["model_id"]) & set(tests["model_b"])


def test_paper_replay_and_s5_summary_are_copied_without_rewriting(tmp_path: Path) -> None:
    paths = build_all(core_root=CORE_ROOT, output_dir=tmp_path)

    assert paths["paper_replay_metrics.csv"].read_bytes() == (
        CORE_ROOT / "output/evaluation/trading_replay_metrics.csv"
    ).read_bytes()
    assert paths["s5_full_summary.json"].read_bytes() == (
        CORE_ROOT / "output/improve_track_evidence/s5_full/s5_full_summary.json"
    ).read_bytes()


def test_manifest_is_complete_and_repeatable(tmp_path: Path) -> None:
    first = build_all(core_root=CORE_ROOT, output_dir=tmp_path)
    first_manifest = first["artifact_manifest.json"].read_bytes()
    manifest = json.loads(first_manifest)

    assert set(manifest["files"]) == EXPECTED_FILES - {"artifact_manifest.json"}
    for name, entry in manifest["files"].items():
        assert entry["sha256"] == sha256(tmp_path / name)
        assert entry["bytes"] == (tmp_path / name).stat().st_size

    second = build_all(core_root=CORE_ROOT, output_dir=tmp_path)
    assert second["artifact_manifest.json"].read_bytes() == first_manifest


def test_generated_primary_artifacts_exclude_retired_names(tmp_path: Path) -> None:
    build_all(core_root=CORE_ROOT, output_dir=tmp_path)

    combined = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in sorted(tmp_path.iterdir())
    ).lower()
    assert "arima" not in combined
    assert "s5-150" not in combined


def test_cli_runs_from_the_repo_root(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            str(CORE_ROOT / ".venv/bin/python"),
            str(CORE_ROOT / "scripts/build_thesis_artifacts.py"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=CORE_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "1824 controlled prediction rows" in completed.stdout
