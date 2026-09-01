"""Create and verify the allow-listed final thesis evidence package."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import tempfile
import uuid


PACKAGE_MAP = {
    "forecast/paper_reported_metrics.csv": "paper_reported_metrics.csv",
    "forecast/controlled_predictions.csv": "controlled_predictions.csv",
    "forecast/controlled_forecast_metrics.csv": "controlled_forecast_metrics.csv",
    "forecast/paired_significance_tests.csv": "paired_significance_tests.csv",
    "trading/paper_replay_metrics.csv": "paper_replay_metrics.csv",
    "trading/corrected_trading_metrics.csv": "corrected_trading_metrics.csv",
    "trading/corrected_trading_equity.csv": "corrected_trading_equity.csv",
    "trading/corrected_trading_metadata.json": "corrected_trading_metadata.json",
    "model/s5_full_summary.json": "s5_full_summary.json",
    "model/checkpoint_provenance.json": "checkpoint_provenance.json",
    "provenance/source_artifact_manifest.json": "artifact_manifest.json",
}
CHECKSUM_NAME = "SHA256SUMS"


@dataclass(frozen=True)
class PackageReceipt:
    root: Path
    file_count: int
    package_sha256: str
    manifest_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_readme(root: Path) -> None:
    (root / "README.md").write_text(
        """# CryptoMamba-v Final Thesis Evidence

This directory is the checksum-verified evidence surface used by the final
thesis and Console. It contains only the approved local comparison:
reproduced CM-v, CMamba-T/S5-Full, and naive persistence.

## Evidence boundaries

- `forecast/paper_reported_metrics.csv`: aggregate values transcribed from
  CryptoMamba v2, Table 3. These rows are labelled `Paper-reported` and are
  not used in paired tests.
- `forecast/controlled_predictions.csv`: locally aligned per-date predictions
  on 304 validation and 304 test dates per model.
- `forecast/controlled_forecast_metrics.csv`: RMSE, MAE, MAPE, direction,
  coverage, parameters, dates, and checkpoint hashes.
- `forecast/paired_significance_tests.csv`: Diebold-Mariano on squared error
  with HAC lag 1 and Wilcoxon signed-rank on absolute error.
- `trading/paper_replay_metrics.csv`: unchanged reproduction of the published
  strategy replay.
- `trading/corrected_trading_*`: separate same-close self-financing evaluation
  with fee-aware sizing and reconciled equity.
- `model/*`: S5-Full result summary and verified checkpoint provenance.
- `provenance/source_artifact_manifest.json`: source-to-output hashes from the
  deterministic core build.

`SHA256SUMS` covers the 13 regular evidence files. Verify it before consuming
any result. Checkpoint binaries are not duplicated here; their paths, sizes,
and SHA-256 values are recorded in `model/checkpoint_provenance.json`.
""",
        encoding="utf-8",
    )


def _write_artifact_map(root: Path) -> None:
    _write_json(
        root / "ARTIFACT_MAP.json",
        {
            "schema_version": 1,
            "source_labels": {
                "paper": "Paper-reported aggregate; no per-date paired test",
                "local": "Locally recomputed from aligned frozen predictions",
            },
            "surfaces": {
                "thesis.paper_reference_table": "forecast/paper_reported_metrics.csv",
                "thesis.controlled_val_test_tables": "forecast/controlled_forecast_metrics.csv",
                "thesis.statistical_tests": "forecast/paired_significance_tests.csv",
                "thesis.paper_trading_replay": "trading/paper_replay_metrics.csv",
                "thesis.corrected_trading": "trading/corrected_trading_metrics.csv",
                "figures.controlled_forecast": "forecast/controlled_forecast_metrics.csv",
                "figures.parameter_error": "forecast/controlled_forecast_metrics.csv",
                "figures.corrected_trading": "trading/corrected_trading_metrics.csv",
                "console.evaluation": "forecast/controlled_forecast_metrics.csv",
                "console.trading": "trading/corrected_trading_metrics.csv",
                "console.architecture": "model/checkpoint_provenance.json",
            },
        },
    )


def _regular_relative_paths(root: Path) -> list[str]:
    paths: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"evidence package contains a symlink: {path}")
        if path.is_file() and path.name != CHECKSUM_NAME:
            paths.append(path.relative_to(root).as_posix())
    return sorted(paths)


def _write_checksums(root: Path) -> Path:
    paths = _regular_relative_paths(root)
    manifest = root / CHECKSUM_NAME
    manifest.write_text(
        "".join(f"{_sha256(root / relative)}  {relative}\n" for relative in paths),
        encoding="utf-8",
    )
    return manifest


def _build_staging(source_dir: Path, staging: Path) -> None:
    for relative_target, source_name in PACKAGE_MAP.items():
        source = source_dir / source_name
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"required generated artifact is missing: {source}")
        target = staging / relative_target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    _write_readme(staging)
    _write_artifact_map(staging)
    _write_checksums(staging)


def package_evidence(source_dir: Path, destination: Path) -> PackageReceipt:
    """Build off-path, verify, then atomically replace only the generated root."""
    source_dir = Path(source_dir).resolve()
    destination = Path(destination).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"artifact source directory is missing: {source_dir}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-build-", dir=destination.parent)
    )
    backup: Path | None = None
    try:
        _build_staging(source_dir, staging)
        staged_receipt = verify_evidence(staging)
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() or not destination.is_dir():
                raise ValueError(f"evidence destination must be a directory: {destination}")
            backup = destination.with_name(
                f".{destination.name}-backup-{uuid.uuid4().hex}"
            )
            destination.rename(backup)
        try:
            staging.rename(destination)
        except BaseException:
            if backup is not None and backup.exists() and not destination.exists():
                backup.rename(destination)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return PackageReceipt(
            root=destination,
            file_count=staged_receipt.file_count,
            package_sha256=staged_receipt.package_sha256,
            manifest_path=destination / CHECKSUM_NAME,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_evidence(destination: Path) -> PackageReceipt:
    root = Path(destination).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"evidence directory is missing: {root}")
    paths = _regular_relative_paths(root)
    manifest_path = root / CHECKSUM_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(f"checksum manifest is missing: {manifest_path}")

    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = raw_line.split("  ", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"invalid checksum line {line_number}")
        expected, relative = parts
        pure = PurePosixPath(relative)
        if (
            len(expected) != 64
            or any(char not in "0123456789abcdef" for char in expected)
        ):
            raise ValueError(f"invalid SHA-256 on line {line_number}")
        if pure.is_absolute() or ".." in pure.parts or relative in {"", "."}:
            raise ValueError(f"unsafe checksum path on line {line_number}: {relative}")
        if relative in entries:
            raise ValueError(f"duplicate checksum path: {relative}")
        entries[relative] = expected

    expected_paths = set(entries)
    actual_paths = set(paths)
    extra = sorted(actual_paths - expected_paths)
    missing = sorted(expected_paths - actual_paths)
    if extra:
        raise ValueError(f"evidence package has unexpected files: {extra}")
    if missing:
        raise ValueError(f"evidence package is missing files: {missing}")
    for relative, expected in entries.items():
        actual = _sha256(root / relative)
        if actual != expected:
            raise ValueError(
                f"checksum mismatch for {relative}: expected {expected}, got {actual}"
            )
    return PackageReceipt(
        root=root,
        file_count=len(entries),
        package_sha256=_sha256(manifest_path),
        manifest_path=manifest_path,
    )
