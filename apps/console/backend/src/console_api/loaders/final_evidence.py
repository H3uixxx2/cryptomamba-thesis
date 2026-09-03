"""Checksum-verified, path-safe reads from the evidence package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

import pandas as pd


_MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  ([^\\]+)$")


class EvidenceIntegrityError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
        or path.as_posix() != value
    ):
        raise EvidenceIntegrityError(f"unsafe evidence path: {value!r}")
    return path


class FinalEvidence:
    """Verify once, then read only files covered by the package manifest."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        manifest = self.root / "SHA256SUMS"
        if not self.root.is_dir() or not manifest.is_file() or manifest.is_symlink():
            raise EvidenceIntegrityError(
                f"final evidence manifest is unavailable under {self.root}"
            )
        try:
            lines = manifest.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise EvidenceIntegrityError("final evidence manifest is unreadable") from exc
        if not lines:
            raise EvidenceIntegrityError("final evidence manifest is empty")

        expected: dict[str, str] = {}
        for line in lines:
            match = _MANIFEST_LINE.fullmatch(line)
            if match is None:
                raise EvidenceIntegrityError(f"invalid SHA256SUMS line: {line!r}")
            digest, relative = match.groups()
            _safe_relative(relative)
            if relative in expected:
                raise EvidenceIntegrityError(f"duplicate manifest path: {relative}")
            expected[relative] = digest

        actual: set[str] = set()
        for path in self.root.rglob("*"):
            if path.is_symlink():
                raise EvidenceIntegrityError(f"symlink is not allowed: {path}")
            if path.is_file() and path != manifest:
                actual.add(path.relative_to(self.root).as_posix())
        if actual != set(expected):
            missing = sorted(set(expected) - actual)
            extra = sorted(actual - set(expected))
            raise EvidenceIntegrityError(
                f"evidence file set mismatch; missing={missing}, extra={extra}"
            )
        for relative, expected_digest in expected.items():
            path = self.root.joinpath(*PurePosixPath(relative).parts)
            actual_digest = _sha256(path)
            if actual_digest != expected_digest:
                raise EvidenceIntegrityError(
                    f"checksum mismatch for {relative}: expected {expected_digest}, "
                    f"received {actual_digest}"
                )

        self._files = frozenset(expected)
        self.package_sha256 = _sha256(manifest)

    def _path(self, relative: str, *, suffix: str) -> Path:
        safe = _safe_relative(relative)
        if safe.suffix.lower() != suffix or safe.as_posix() not in self._files:
            raise EvidenceIntegrityError(
                f"unlisted or invalid evidence file: {relative!r}"
            )
        return self.root.joinpath(*safe.parts)

    def read_csv(self, relative: str) -> pd.DataFrame:
        path = self._path(relative, suffix=".csv")
        try:
            return pd.read_csv(path)
        except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise EvidenceIntegrityError(f"evidence CSV is unreadable: {relative}") from exc

    def read_json(self, relative: str) -> Any:
        path = self._path(relative, suffix=".json")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvidenceIntegrityError(f"evidence JSON is unreadable: {relative}") from exc


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a frame to standards-compliant JSON records (NaN -> null)."""
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


__all__ = ["EvidenceIntegrityError", "FinalEvidence", "records"]
