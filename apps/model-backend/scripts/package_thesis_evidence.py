"""Build or verify the clean final CryptoMamba-v evidence directory."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


CORE_ROOT = Path(__file__).resolve().parents[1]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from thesis_pipeline.package import package_evidence, verify_evidence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=CORE_ROOT / "output/thesis_final",
        help="Generated final artifact directory.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=CORE_ROOT.parent / "cryptomamba-thesis-evidence/final",
        help="Allow-listed evidence package root.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing destination without changing it.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    receipt = (
        verify_evidence(args.destination)
        if args.verify_only
        else package_evidence(args.source, args.destination)
    )
    verb = "Verified" if args.verify_only else "Packaged"
    print(
        f"{verb} evidence package: {receipt.file_count} files; "
        f"SHA-256 {receipt.package_sha256}; root {receipt.root}"
    )


if __name__ == "__main__":
    main()
