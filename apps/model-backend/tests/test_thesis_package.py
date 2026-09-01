from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from scripts.build_thesis_artifacts import build_all
from thesis_pipeline.package import package_evidence, verify_evidence


CORE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_FILES = {
    "README.md",
    "ARTIFACT_MAP.json",
    "forecast/paper_reported_metrics.csv",
    "forecast/controlled_predictions.csv",
    "forecast/controlled_forecast_metrics.csv",
    "forecast/paired_significance_tests.csv",
    "trading/paper_replay_metrics.csv",
    "trading/corrected_trading_metrics.csv",
    "trading/corrected_trading_equity.csv",
    "trading/corrected_trading_metadata.json",
    "model/s5_full_summary.json",
    "model/checkpoint_provenance.json",
    "provenance/source_artifact_manifest.json",
}


@pytest.fixture(scope="module")
def source_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("thesis-source")
    build_all(core_root=CORE_ROOT, output_dir=root)
    return root


def relative_regular_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def test_package_contains_only_the_approved_allow_list(
    source_dir: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "final"

    receipt = package_evidence(source_dir, destination)

    assert relative_regular_files(destination) == PACKAGE_FILES | {"SHA256SUMS"}
    assert receipt.file_count == len(PACKAGE_FILES)
    assert receipt.manifest_path == destination / "SHA256SUMS"
    names_and_text = "\n".join(
        [
            *(sorted(relative_regular_files(destination))),
            *(
                path.read_text(encoding="utf-8", errors="ignore")
                for path in destination.rglob("*")
                if path.is_file()
            ),
        ]
    ).lower()
    assert "arima" not in names_and_text
    assert "s5-150" not in names_and_text


def test_manifest_verifies_every_regular_file(
    source_dir: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "final"
    receipt = package_evidence(source_dir, destination)

    verified = verify_evidence(destination)

    assert verified.file_count == receipt.file_count
    assert verified.package_sha256 == receipt.package_sha256


def test_verifier_rejects_tampering(source_dir: Path, tmp_path: Path) -> None:
    destination = tmp_path / "final"
    package_evidence(source_dir, destination)
    metrics = destination / "forecast/controlled_forecast_metrics.csv"
    metrics.write_text(metrics.read_text() + "tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_evidence(destination)


def test_verifier_rejects_extra_and_symlinked_files(
    source_dir: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "final"
    package_evidence(source_dir, destination)
    (destination / "unexpected.txt").write_text("extra", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected files"):
        verify_evidence(destination)

    (destination / "unexpected.txt").unlink()
    (destination / "link.csv").symlink_to(
        destination / "forecast/controlled_forecast_metrics.csv"
    )
    with pytest.raises(ValueError, match="symlink"):
        verify_evidence(destination)


def test_cli_packages_and_verifies_from_repo_root(
    source_dir: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "final"
    command = [
        str(CORE_ROOT / ".venv/bin/python"),
        str(CORE_ROOT / "scripts/package_thesis_evidence.py"),
        "--source",
        str(source_dir),
        "--destination",
        str(destination),
    ]

    built = subprocess.run(
        command,
        cwd=CORE_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    verified = subprocess.run(
        [
            *command[:2],
            "--verify-only",
            "--destination",
            str(destination),
        ],
        cwd=CORE_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert built.returncode == 0, built.stderr
    assert verified.returncode == 0, verified.stderr
    assert "13 files" in built.stdout
    assert "13 files" in verified.stdout
