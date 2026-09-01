from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume"]
PAPER_SPLITS: dict[str, tuple[str, str]] = {
    "train": ("2018-09-17", "2022-09-17"),
    "validation": ("2022-09-17", "2023-09-17"),
    "test": ("2023-09-17", "2024-09-17"),
}
SPLIT_ORDER = ["train", "validation", "test", "out_of_scope"]
MODEL_FEATURE_ORDER = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
MODEL_WINDOW_SIZE = 14
# Last paper-split date the model was trained/evaluated on. Input windows ending on
# or after this are extrapolation: the checkpoint never saw this price/timestamp
# regime (normalize=False, raw price + raw Timestamp features), so predictions there
# are a qualitative demo, not validated research evidence.
MODEL_TRAIN_HORIZON = PAPER_SPLITS["test"][1]  # "2024-09-17"

_COLUMN_ALIASES = {
    "date": "date",
    "datetime": "date",
    "day": "date",
    "timestamp": "timestamp",
    "time": "timestamp",
    "timeopen": "date",
    "opentime": "timestamp",
    "openat": "date",
    "starttime": "date",
    "startdate": "date",
    "open": "open",
    "openprice": "open",
    "priceopen": "open",
    "high": "high",
    "highprice": "high",
    "pricehigh": "high",
    "low": "low",
    "lowprice": "low",
    "pricelow": "low",
    "close": "close",
    "closeprice": "close",
    "priceclose": "close",
    "volume": "volume",
    "vol": "volume",
    "volumeusd": "volume",
    "volume24h": "volume",
}

_DATE_COLUMN_PRIORITY = [
    "date",
    "day",
    "timeopen",
    "opentime",
    "openat",
    "starttime",
    "startdate",
    "datetime",
    "timestamp",
    "time",
]


class CandleDataError(ValueError):
    """Raised when uploaded candle data is invalid."""


def load_csv(path_or_buffer: str | Path | Any) -> pd.DataFrame:
    return normalize_candles(pd.read_csv(path_or_buffer))


def normalize_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize uploaded OHLCV rows into sorted daily-compatible candle rows.

    This function intentionally fails fast on malformed uploaded data instead of
    silently dropping bad rows. A thesis/demo user should know why input data is
    rejected before the app builds a model payload from it.
    """
    if df.empty:
        raise CandleDataError("CSV is empty")

    source_columns = {target: _find_column(df, target) for target in REQUIRED_COLUMNS}
    missing = [column for column, source_column in source_columns.items() if source_column is None]
    if missing:
        detected = ", ".join(str(column) for column in df.columns[:12])
        suffix = "..." if len(df.columns) > 12 else ""
        raise CandleDataError(
            f"Missing required columns: {', '.join(missing)}. "
            f"Detected columns: {detected}{suffix}"
        )

    parsed_timestamps = _parse_timestamps(df[source_columns["date"]])
    out = pd.DataFrame(index=df.index)
    out["date"] = parsed_timestamps.dt.strftime("%Y-%m-%d")
    out["_timestamp_order"] = parsed_timestamps
    invalid_dates = int(out["date"].isna().sum())
    if invalid_dates:
        raise CandleDataError(f"Invalid date/timestamp rows: {invalid_dates}")

    invalid_numeric: dict[str, int] = {}
    for column in NUMERIC_COLUMNS:
        out[column] = df[source_columns[column]]
        out[column] = pd.to_numeric(out[column], errors="coerce")
        invalid_count = int((~np.isfinite(out[column].astype(float))).sum())
        if invalid_count:
            invalid_numeric[column] = invalid_count
    if invalid_numeric:
        details = ", ".join(f"{column}={count}" for column, count in invalid_numeric.items())
        raise CandleDataError(f"Invalid numeric OHLCV values: {details}")

    non_positive_price_rows = int((out[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    if non_positive_price_rows:
        raise CandleDataError(f"OHLC price values must be positive. Invalid rows: {non_positive_price_rows}")

    negative_volume_rows = int((out["volume"] < 0).sum())
    if negative_volume_rows:
        raise CandleDataError(f"Volume must be non-negative. Invalid rows: {negative_volume_rows}")

    invalid_ohlc_rows = int(
        (
            (out["high"] < out[["open", "close", "low"]].max(axis=1))
            | (out["low"] > out[["open", "close", "high"]].min(axis=1))
        ).sum()
    )
    if invalid_ohlc_rows:
        raise CandleDataError(
            "Invalid OHLC candle invariants: high must be the maximum and low "
            f"must be the minimum. Invalid rows: {invalid_ohlc_rows}"
        )

    out = (
        out.sort_values("_timestamp_order", kind="stable")
        .drop(columns="_timestamp_order")
        .reset_index(drop=True)
    )
    if len(out) < MODEL_WINDOW_SIZE:
        raise CandleDataError(f"Need at least {MODEL_WINDOW_SIZE} valid daily candles")
    return out


def _canonical_column_key(column: object) -> str:
    return "".join(char for char in str(column).strip().lower() if char.isalnum())


def _find_column(df: pd.DataFrame, target: str) -> Any | None:
    keyed_columns = [(_canonical_column_key(column), column) for column in df.columns]
    if target == "date":
        for preferred_key in _DATE_COLUMN_PRIORITY:
            for key, column in keyed_columns:
                if key == preferred_key:
                    return column
    for key, column in keyed_columns:
        if _COLUMN_ALIASES.get(key) == target:
            return column
    if target == "date":
        return _infer_parseable_date_column(df, keyed_columns)
    return None


def _infer_parseable_date_column(df: pd.DataFrame, keyed_columns: list[tuple[str, Any]]) -> Any | None:
    for key, column in keyed_columns:
        if not any(token in key for token in ("date", "time", "timestamp")):
            continue
        values = df[column]
        non_null = int(values.notna().sum())
        if non_null == 0:
            continue
        parsed = _parse_timestamp_or_date(values)
        if int(parsed.notna().sum()) == non_null:
            return column
    return None


def _parse_timestamp_or_date(values: pd.Series) -> pd.Series:
    return _parse_timestamps(values).dt.strftime("%Y-%m-%d")


def _parse_timestamps(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    non_null = values.notna().sum()
    if numeric.notna().sum() == non_null and non_null > 0:
        median = numeric.dropna().median()
        unit = "ms" if median > 10_000_000_000 else "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce")
    return pd.to_datetime(values, errors="coerce", utc=True)


def daily_ohlcv(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate normalized rows into one OHLCV candle per date."""
    normalized = normalize_candles(raw_df)
    return (
        normalized.groupby("date", sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )


def add_dataset_split(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        raise CandleDataError("Missing required columns: date")

    out = df.copy()
    dates = pd.to_datetime(out["date"], errors="coerce")
    if dates.isna().any():
        raise CandleDataError(f"Invalid date rows before split: {int(dates.isna().sum())}")

    out["dataset_split"] = "out_of_scope"
    for split_name, (start, end) in PAPER_SPLITS.items():
        out.loc[(dates >= start) & (dates < end), "dataset_split"] = split_name
    return out


def add_chronological_split(df: pd.DataFrame, train_ratio: float = 0.70, validation_ratio: float = 0.15) -> pd.DataFrame:
    """Split arbitrary uploaded data chronologically by ratio.

    Paper-date splits are only valid for the paper BTC dataset. Uploaded/custom
    datasets need a split derived from their own date range.
    """
    if "date" not in df.columns:
        raise CandleDataError("Missing required columns: date")
    if not 0 < train_ratio < 1:
        raise CandleDataError("train_ratio must be between 0 and 1")
    if not 0 < validation_ratio < 1:
        raise CandleDataError("validation_ratio must be between 0 and 1")
    if train_ratio + validation_ratio >= 1:
        raise CandleDataError("train_ratio + validation_ratio must be < 1")

    out = df.copy()
    dates = pd.to_datetime(out["date"], errors="coerce")
    if dates.isna().any():
        raise CandleDataError(f"Invalid date rows before split: {int(dates.isna().sum())}")

    out = out.assign(_sort_date=dates).sort_values("_sort_date").drop(columns="_sort_date").reset_index(drop=True)
    row_count = len(out)
    if row_count < 3:
        raise CandleDataError("Need at least 3 rows to build train/validation/test split")

    train_end = max(1, int(row_count * train_ratio))
    validation_end = max(train_end + 1, int(row_count * (train_ratio + validation_ratio)))
    validation_end = min(validation_end, row_count - 1)
    train_end = min(train_end, validation_end - 1)

    out["dataset_split"] = "test"
    out.loc[: train_end - 1, "dataset_split"] = "train"
    out.loc[train_end : validation_end - 1, "dataset_split"] = "validation"
    return out


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    split_names = [split_name for split_name in SPLIT_ORDER if split_name in set(df["dataset_split"])]
    for split_name in split_names:
        part = df[df["dataset_split"] == split_name]
        rows.append(
            {
                "split": split_name,
                "rows": len(part),
                "from": part["date"].min() if not part.empty else "—",
                "to": part["date"].max() if not part.empty else "—",
            }
        )
    return pd.DataFrame(rows)


def build_predict_payload(df: pd.DataFrame, prediction_date: str | None, risk: float) -> dict[str, Any]:
    # The payload window MUST be the candles the chart shows for this date: for a past
    # prediction_date use the window ending strictly before it (same as select_window),
    # otherwise the model would be fed the latest 14 candles while the UI charts a
    # historical window. For the default next-day case, use the last 14 candles.
    if prediction_date:
        normalized = select_window(df, prediction_date)
    else:
        normalized = normalize_candles(df).tail(MODEL_WINDOW_SIZE)
        last_date = datetime.strptime(str(normalized["date"].iloc[-1]), "%Y-%m-%d")
        prediction_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")

    candles = [
        {
            "date": str(row.date),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
        }
        for row in normalized.itertuples(index=False)
    ]
    return {"prediction_date": prediction_date, "risk": float(risk), "candles": candles}


def prediction_date_bounds(df: pd.DataFrame) -> tuple[date, date]:
    """Valid [min, max] prediction dates for the model given a dataset.

    The model needs exactly MODEL_WINDOW_SIZE candles strictly before the target
    date. So the earliest predictable date is the (window+1)-th candle's date, and
    the latest is the day after the last candle (the true next-day forecast).
    """
    normalized = normalize_candles(df)
    dates = pd.to_datetime(normalized["date"])
    max_pred = (dates.iloc[-1] + pd.Timedelta(days=1)).date()
    if len(normalized) > MODEL_WINDOW_SIZE:
        min_pred = dates.iloc[MODEL_WINDOW_SIZE].date()
    else:
        min_pred = max_pred
    return min_pred, max_pred


def select_window(df: pd.DataFrame, prediction_date: str) -> pd.DataFrame:
    """Return the MODEL_WINDOW_SIZE candles ending strictly before prediction_date.

    This slides the fixed-size input window so the UI can predict any date the
    dataset actually supports, instead of always using the last 14 candles.
    """
    normalized = normalize_candles(df)
    cutoff = pd.to_datetime(prediction_date)
    window = normalized[pd.to_datetime(normalized["date"]) < cutoff].tail(MODEL_WINDOW_SIZE)
    if len(window) < MODEL_WINDOW_SIZE:
        raise CandleDataError(
            f"Not enough candles before {prediction_date}: "
            f"need {MODEL_WINDOW_SIZE}, found {len(window)}"
        )
    return window.reset_index(drop=True)


def is_out_of_distribution(window: pd.DataFrame) -> bool:
    """True when the input window ends on/after the model's training/eval horizon.

    Predictions on such windows are extrapolation beyond the validated paper split
    and must be presented as a qualitative demo, never as validated evidence.
    """
    if window is None or window.empty or "date" not in window.columns:
        return False
    last_date = pd.to_datetime(window["date"].iloc[-1])
    return last_date >= pd.to_datetime(MODEL_TRAIN_HORIZON)


def window_from_candles(candles: list[dict[str, Any]]) -> pd.DataFrame:
    """Rebuild the normalized input-window DataFrame from artifact payload candles.

    Lets the offline view chart the frozen prediction against ITS OWN window, not
    the current UI dataset (which need not match the frozen fixture).
    """
    return normalize_candles(pd.DataFrame(candles))


def model_tensor_preview(source_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a UI preview of the real cmamba_v feature tensor contract.

    Repo reference: DataTransform uses feature-major shape [features, seq_len], then
    inference calls model(x[None, ...]), so batch shape is [1, 6, 14].
    """
    normalized = normalize_candles(source_df)
    if len(normalized) < MODEL_WINDOW_SIZE:
        raise CandleDataError(f"Need at least {MODEL_WINDOW_SIZE} valid daily candles")

    window = normalized.tail(MODEL_WINDOW_SIZE).copy()
    dates = pd.to_datetime(window["date"])
    preview = pd.DataFrame(
        {
            "t": range(1, MODEL_WINDOW_SIZE + 1),
            "date": window["date"].astype(str).tolist(),
            "timestamp": (dates.astype("int64") // 10**9).astype(int).tolist(),
            "open": window["open"].astype(float).round(6).tolist(),
            "high": window["high"].astype(float).round(6).tolist(),
            "low": window["low"].astype(float).round(6).tolist(),
            "close": window["close"].astype(float).round(6).tolist(),
            "volume_scaled": (window["volume"].astype(float) / 1e9).round(6).tolist(),
        }
    )
    feature_matrix = [
        preview["timestamp"].tolist(),
        preview["open"].tolist(),
        preview["high"].tolist(),
        preview["low"].tolist(),
        preview["close"].tolist(),
        preview["volume_scaled"].tolist(),
    ]
    payload = {
        "model_config": "cmamba_v",
        "feature_order": MODEL_FEATURE_ORDER,
        "seq_len": MODEL_WINDOW_SIZE,
        "tensor_shape_before_batch": [6, MODEL_WINDOW_SIZE],
        "tensor_shape_sent_to_model": [1, 6, MODEL_WINDOW_SIZE],
        "volume_rule": "Volume / 1e9",
        "normalize": False,
        "features": feature_matrix,
    }
    return preview, payload
