"""Create and verify the allow-listed evidence package (SHA256SUMS + manifest)."""
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
        """# Evidence bundle

The checksum-verified result surface the thesis tables and the console read from. It holds only
the three models with per-date local results: reproduced CryptoMamba-v, CMamba-T / S5-Full, and
naive persistence.

```bash
shasum -c SHA256SUMS      # 13 files
```

Verify before consuming any value. The console verifies on startup and reports `NOT_READY` on any
mismatch rather than serving unverified data.

## Contents

| File | What it is |
|---|---|
| `forecast/controlled_predictions.csv` | Per-date predictions on the 304 validation and 304 test dates the three models share |
| `forecast/controlled_forecast_metrics.csv` | RMSE, MAE, MAPE, directional accuracy, coverage, parameter count, date range, checkpoint hash |
| `forecast/paired_significance_tests.csv` | Diebold–Mariano on squared error (HAC lag 1) and Wilcoxon signed-rank on absolute error |
| `forecast/paper_reported_metrics.csv` | Aggregate values transcribed from CryptoMamba, Table 3. Labelled `Paper-reported`. |
| `trading/paper_replay_metrics.csv` | Unchanged reproduction of the published strategy replay |
| `trading/corrected_trading_metrics.csv` · `corrected_trading_equity.csv` · `corrected_trading_metadata.json` | Separate same-close self-financing evaluation: fee-aware sizing, reconciled equity |
| `model/s5_full_summary.json` | S5-Full run summary |
| `model/checkpoint_provenance.json` | Both checkpoints by path, byte count and SHA-256 |
| `provenance/source_artifact_manifest.json` | Source-to-output hashes from the deterministic build |
| `ARTIFACT_MAP.json` | Which file backs which thesis table and which console screen |

## Boundaries

- The `paper_reported` rows have no per-date series and therefore never enter a paired test. They
  are a published reference, not a locally reproduced result.
- The 304-date sets differ from the 350-date reproduction protocol in step 1 of
  `docs/reproduce.md`. They are not interchangeable and are never pooled.
- `trading/paper_replay_metrics.csv` and `trading/corrected_trading_*` use different accounting
  rules — zero-fee published replay versus fee-aware self-financing — so their balances are not
  directly comparable.
- Checkpoint binaries are not duplicated here. `model/checkpoint_provenance.json` records their
  paths, sizes and SHA-256 values under `apps/model-backend/output/`.

Regenerate with `scripts/build_thesis_artifacts.py` followed by
`scripts/package_thesis_evidence.py` from `apps/model-backend/`.
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
