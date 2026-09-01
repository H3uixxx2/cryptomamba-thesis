from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - repository normally has PyYAML via existing scripts
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_CONFIG = ROOT / "configs" / "training" / "cmamba_v.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "evaluation"
SPLITS = ("train", "val", "test")
REQUIRED_COLUMNS = ("Timestamp", "Open", "High", "Low", "Close", "Volume")
DEFAULT_WINDOW_SIZE = 14
DEFAULT_HORIZON_DAYS = 1


@dataclass(frozen=True)
class EvaluationConfig:
    data_dir: Path
    output_dir: Path
    window_size: int
    horizon_days: int
    training_config: Path | None
    source_sample_count_mode: str


@dataclass(frozen=True)
class SplitQuality:
    split: str
    rows: int
    samples: int
    start_timestamp: int
    end_timestamp: int
    start_date: str
    end_date: str
    duplicate_timestamps: int
    null_cells: int
    non_positive_close_rows: int
    monotonic_timestamp: bool
    expected_step_seconds: int | None
    step_violations: int


@dataclass(frozen=True)
class BaselineSpec:
    model: str
    description: str


BASELINES = (
    BaselineSpec(
        model="naive_persistence",
        description="Predict next close as the latest available close: y_hat[t+1] = close[t].",
    ),
)


class BaselineEvaluationError(RuntimeError):
    """Raised when baseline evaluation cannot produce trustworthy artifacts."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate deterministic forecasting baselines on the CryptoMamba paper splits "
            "and write CSV/JSON artifacts for thesis/UI consumption."
        )
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG,
        help="Training YAML used to resolve the data_config. Pass --data-dir to bypass YAML resolution.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing train.csv, val.csv, and test.csv. Overrides --training-config data_config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where baseline_metrics.csv, baseline_predictions.csv, data_quality.csv, and metadata JSON are written.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Model input window size. CryptoMamba-v released setup uses 14.",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=DEFAULT_HORIZON_DAYS,
        help="Forecast horizon in daily rows. Only 1 is currently supported to match the paper setup.",
    )
    parser.add_argument(
        "--sample-count-mode",
        choices=("released", "complete"),
        default="released",
        help=(
            "released: match CMambaDataset.__len__ = len(split)-window-1, which excludes the final possible target; "
            "complete: use all len(split)-window one-step targets."
        ),
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise BaselineEvaluationError("PyYAML is required when resolving --training-config. Pass --data-dir to bypass YAML.")
    if not path.exists():
        raise FileNotFoundError(f"Missing training config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise BaselineEvaluationError(f"Training config must be a mapping: {path}")
    return data


def resolve_data_dir(training_config: Path, explicit_data_dir: Path | None) -> Path:
    if explicit_data_dir is not None:
        return explicit_data_dir.resolve()

    training = load_yaml(training_config)
    data_config_name = training.get("data_config")
    if not data_config_name:
        raise BaselineEvaluationError(f"Missing data_config in {training_config}")

    data_config_path = ROOT / "configs" / "data_configs" / f"{data_config_name}.yaml"
    data_config = load_yaml(data_config_path)
    root = Path(str(data_config.get("root", ROOT / "data")))
    if not root.is_absolute():
        root = ROOT / root

    required_keys = ("start_date", "end_date", "jumps")
    missing = [key for key in required_keys if data_config.get(key) is None]
    if missing:
        raise BaselineEvaluationError(f"{data_config_path} missing required keys: {missing}")

    folder_name = f"{data_config['start_date']}_{data_config['end_date']}_{data_config['jumps']}"
    return (root / folder_name).resolve()


def read_split(data_dir: Path, split: str) -> pd.DataFrame:
    path = data_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    df = pd.read_csv(path, index_col=0)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise BaselineEvaluationError(f"{path} missing required columns: {missing}")

    df = df.loc[:, REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        null_counts = df[list(REQUIRED_COLUMNS)].isna().sum().to_dict()
        raise BaselineEvaluationError(f"{path} contains non-numeric/null required values: {null_counts}")

    df = df.sort_values("Timestamp").reset_index(drop=True)
    df["Timestamp"] = df["Timestamp"].astype(int)
    df["date"] = pd.to_datetime(df["Timestamp"], unit="s").dt.strftime("%Y-%m-%d")
    return df


def configured_step_seconds(training_config: Path | None) -> int | None:
    if training_config is None:
        return None
    training = load_yaml(training_config)
    data_config_name = training.get("data_config")
    if not data_config_name:
        return None
    data_config_path = ROOT / "configs" / "data_configs" / f"{data_config_name}.yaml"
    data_config = load_yaml(data_config_path)
    jumps = data_config.get("jumps")
    if jumps is None:
        return None
    jumps = int(jumps)
    # Historical generated config.pkl files in this repo may store daily jumps as 1440 minutes.
    # The canonical training data_config uses seconds. Normalize defensively.
    return jumps * 60 if jumps < 10_000 else jumps


def sample_count(row_count: int, window_size: int, mode: str) -> int:
    if mode == "released":
        # Mirrors data_utils.dataset.CMambaDataset.__len__ exactly.
        return max(0, row_count - window_size - 1)
    if mode == "complete":
        return max(0, row_count - window_size)
    raise ValueError(f"Unsupported sample-count mode: {mode}")


def validate_split(df: pd.DataFrame, split: str, window_size: int, mode: str, step_seconds: int | None) -> SplitQuality:
    if len(df) <= window_size:
        raise BaselineEvaluationError(f"{split} has {len(df)} rows, not enough for window_size={window_size}")
    if (df["Close"] <= 0).any():
        raise BaselineEvaluationError(f"{split} contains non-positive Close values")

    diffs = df["Timestamp"].diff().dropna()
    inferred_step = int(diffs.mode().iloc[0]) if not diffs.empty else step_seconds
    expected_step = step_seconds or inferred_step
    step_violations = int((diffs != expected_step).sum()) if expected_step is not None else 0
    if expected_step is not None and step_violations:
        raise BaselineEvaluationError(f"{split} has {step_violations} timestamp step violations; expected {expected_step}s")

    return SplitQuality(
        split=split,
        rows=int(len(df)),
        samples=sample_count(len(df), window_size, mode),
        start_timestamp=int(df["Timestamp"].iloc[0]),
        end_timestamp=int(df["Timestamp"].iloc[-1]),
        start_date=str(df["date"].iloc[0]),
        end_date=str(df["date"].iloc[-1]),
        duplicate_timestamps=int(df["Timestamp"].duplicated().sum()),
        null_cells=int(df[list(REQUIRED_COLUMNS)].isna().sum().sum()),
        non_positive_close_rows=int((df["Close"] <= 0).sum()),
        monotonic_timestamp=bool(df["Timestamp"].is_monotonic_increasing),
        expected_step_seconds=expected_step,
        step_violations=step_violations,
    )


def build_naive_persistence_predictions(df: pd.DataFrame, split: str, config: EvaluationConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_samples = sample_count(len(df), config.window_size, config.source_sample_count_mode)
    for sample_index in range(n_samples):
        current_idx = sample_index + config.window_size - 1
        target_idx = current_idx + config.horizon_days
        if target_idx >= len(df):
            break

        current_close = float(df.loc[current_idx, "Close"])
        target_close = float(df.loc[target_idx, "Close"])
        predicted_close = current_close
        rows.append(
            {
                "model": "naive_persistence",
                "split": split,
                "sample_index": sample_index,
                "window_start_date": df.loc[sample_index, "date"],
                "window_end_date": df.loc[current_idx, "date"],
                "prediction_date": df.loc[target_idx, "date"],
                "window_start_timestamp": int(df.loc[sample_index, "Timestamp"]),
                "window_end_timestamp": int(df.loc[current_idx, "Timestamp"]),
                "prediction_timestamp": int(df.loc[target_idx, "Timestamp"]),
                "current_close": current_close,
                "target_close": target_close,
                "predicted_close": predicted_close,
                "actual_return_pct": ((target_close - current_close) / current_close) * 100.0,
                "predicted_return_pct": ((predicted_close - current_close) / current_close) * 100.0,
                "absolute_error": abs(predicted_close - target_close),
                "squared_error": (predicted_close - target_close) ** 2,
            }
        )
    return pd.DataFrame(rows)


def direction_metrics(group: pd.DataFrame) -> dict[str, float]:
    actual_direction = np.sign(group["target_close"].to_numpy(dtype=float) - group["current_close"].to_numpy(dtype=float))
    predicted_direction = np.sign(group["predicted_close"].to_numpy(dtype=float) - group["current_close"].to_numpy(dtype=float))

    strict_hit = float(np.mean(actual_direction == predicted_direction) * 100.0)
    non_hold_mask = predicted_direction != 0
    coverage = float(np.mean(non_hold_mask) * 100.0)
    when_predicting = float(np.mean(actual_direction[non_hold_mask] == predicted_direction[non_hold_mask]) * 100.0) if np.any(non_hold_mask) else math.nan
    return {
        "directional_accuracy_strict_pct": strict_hit,
        "directional_coverage_pct": coverage,
        "directional_accuracy_when_predicting_pct": when_predicting,
    }


def metric_rows(predictions: pd.DataFrame, config: EvaluationConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, split), group in predictions.groupby(["model", "split"], sort=False):
        target = group["target_close"].to_numpy(dtype=float)
        pred = group["predicted_close"].to_numpy(dtype=float)
        error = pred - target
        nonzero_target = target != 0
        mse = float(np.mean(error**2))
        mape = float(np.mean(np.abs(error[nonzero_target] / target[nonzero_target])) * 100.0) if np.any(nonzero_target) else math.nan
        dm = direction_metrics(group)
        rows.append(
            {
                "model": model,
                "split": split,
                "samples": int(len(group)),
                "window_size": config.window_size,
                "forecast_horizon_days": config.horizon_days,
                "target": "Close",
                "MSE": round(mse, 6),
                "RMSE": round(float(np.sqrt(mse)), 6),
                "MAE": round(float(np.mean(np.abs(error))), 6),
                "MAPE_pct": round(mape, 6) if not math.isnan(mape) else math.nan,
                "bias_mean_error": round(float(np.mean(error)), 6),
                "median_absolute_error": round(float(np.median(np.abs(error))), 6),
                "directional_accuracy_strict_pct": round(dm["directional_accuracy_strict_pct"], 6),
                "directional_coverage_pct": round(dm["directional_coverage_pct"], 6),
                "directional_accuracy_when_predicting_pct": (
                    round(dm["directional_accuracy_when_predicting_pct"], 6)
                    if not math.isnan(dm["directional_accuracy_when_predicting_pct"])
                    else math.nan
                ),
                "metric_note": (
                    "naive_persistence predicts no price movement, so directional coverage is 0%; "
                    "strict directional accuracy counts HOLD as a direction and should not be overinterpreted."
                ),
                "protocol": f"{config.source_sample_count_mode}_sample_count; y_hat[t+1]=close[t]",
            }
        )
    return pd.DataFrame(rows)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def git_dirty() -> bool | None:
    try:
        return bool(subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip())
    except Exception:
        return None


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_artifacts(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    quality: Iterable[SplitQuality],
    config: EvaluationConfig,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(metrics, config.output_dir / "baseline_metrics.csv")
    atomic_write_csv(predictions, config.output_dir / "baseline_predictions.csv")
    atomic_write_csv(pd.DataFrame([asdict(item) for item in quality]), config.output_dir / "data_quality.csv")

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(ROOT),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "config": {
            "data_dir": str(config.data_dir),
            "output_dir": str(config.output_dir),
            "window_size": config.window_size,
            "horizon_days": config.horizon_days,
            "training_config": str(config.training_config) if config.training_config else None,
            "source_sample_count_mode": config.source_sample_count_mode,
        },
        "baselines": [asdict(spec) for spec in BASELINES],
        "metric_definitions": {
            "RMSE": "sqrt(mean((predicted_close - target_close)^2))",
            "MAE": "mean(abs(predicted_close - target_close))",
            "MAPE_pct": "mean(abs(error / target_close)) * 100; zero targets excluded",
            "directional_accuracy_strict_pct": "mean(sign(predicted_close-current_close) == sign(target_close-current_close)) * 100",
            "directional_coverage_pct": "percent of rows where predicted direction is non-HOLD",
            "directional_accuracy_when_predicting_pct": "directional accuracy over non-HOLD predictions only; NaN when coverage is 0",
        },
    }
    atomic_write_json(metadata, config.output_dir / "baseline_metadata.json")


def main() -> None:
    args = parse_args()
    if args.window_size < 1:
        raise ValueError("--window-size must be positive")
    if args.horizon_days != 1:
        raise NotImplementedError("Only --horizon-days 1 is supported for paper-aligned baseline evaluation")

    training_config = args.training_config.resolve() if args.training_config else None
    data_dir = resolve_data_dir(training_config, args.data_dir) if training_config else args.data_dir.resolve()
    if data_dir is None:
        raise BaselineEvaluationError("Unable to resolve data directory")

    config = EvaluationConfig(
        data_dir=data_dir,
        output_dir=args.output_dir.resolve(),
        window_size=args.window_size,
        horizon_days=args.horizon_days,
        training_config=training_config,
        source_sample_count_mode=args.sample_count_mode,
    )

    step_seconds = configured_step_seconds(training_config)
    split_frames: dict[str, pd.DataFrame] = {split: read_split(config.data_dir, split) for split in SPLITS}
    quality = [validate_split(df, split, config.window_size, config.source_sample_count_mode, step_seconds) for split, df in split_frames.items()]

    predictions = pd.concat(
        [build_naive_persistence_predictions(df, split, config) for split, df in split_frames.items()],
        ignore_index=True,
    )
    if predictions.empty:
        raise BaselineEvaluationError("No predictions generated")

    metrics = metric_rows(predictions, config)
    write_artifacts(predictions, metrics, quality, config)

    print(f"Wrote {config.output_dir / 'baseline_metrics.csv'}")
    print(f"Wrote {config.output_dir / 'baseline_predictions.csv'}")
    print(f"Wrote {config.output_dir / 'data_quality.csv'}")
    print(f"Wrote {config.output_dir / 'baseline_metadata.json'}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
