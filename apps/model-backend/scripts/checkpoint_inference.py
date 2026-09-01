#!/usr/bin/env python3
"""Read one checkpoint-inference request from stdin and emit one JSON response."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys


CORE_ROOT = Path(__file__).resolve().parents[1]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from thesis_pipeline.inference import (  # noqa: E402
    PredictionValidationError,
    parse_request,
    predict_next_close,
)


def _write_error(error_type: str, message: str) -> None:
    print(
        json.dumps(
            {"error_type": error_type, "message": message},
            allow_nan=False,
            separators=(",", ":"),
        ),
        file=sys.stderr,
    )


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        request = parse_request(payload)
        result = predict_next_close(request)
    except json.JSONDecodeError as exc:
        _write_error("validation", f"request is not valid JSON: {exc.msg}")
        return 2
    except PredictionValidationError as exc:
        _write_error("validation", str(exc))
        return 2
    except FileNotFoundError as exc:
        _write_error("checkpoint_unavailable", str(exc))
        return 3
    except RuntimeError as exc:
        _write_error("inference_failure", str(exc))
        return 4

    print(json.dumps(asdict(result), allow_nan=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
