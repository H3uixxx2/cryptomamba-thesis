from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from console_api.vendor.cryptomamba_ui.data import (
    MODEL_WINDOW_SIZE,
    CandleDataError,
    add_chronological_split,
    add_dataset_split,
    daily_ohlcv,
)


@dataclass(frozen=True)
class DatasetBundle:
    """Processed dataset state used by the Streamlit pages."""

    raw_df: pd.DataFrame
    daily_df: pd.DataFrame
    processed_df: pd.DataFrame
    source_label: str
    source_detail: str
    split_strategy: str


class DatasetService:
    """Load and process datasets for the app.

    UI code owns file selection/rendering. This service owns IO orchestration:
    read raw CSV -> normalize/aggregate daily OHLCV -> apply thesis split.
    Pure validation/transform rules stay in data.py.
    """

    def __init__(self, sample_path: Path, display_root: Path) -> None:
        self._sample_path = sample_path
        self._display_root = display_root

    def load_paper_sample(self) -> DatasetBundle:
        return self._process_csv(
            self._sample_path,
            source_label="Paper dataset",
            source_detail=f"File: {self._sample_path.relative_to(self._display_root)}",
            split_strategy="paper_date",
        )

    def load_uploaded_csv(self, uploaded_file: Any, filename: str | None = None) -> DatasetBundle:
        name = filename or getattr(uploaded_file, "name", "uploaded.csv")
        return self._process_csv(
            uploaded_file,
            source_label="Uploaded CSV",
            source_detail=f"File: {name}",
            split_strategy="chronological_ratio",
        )

    @staticmethod
    def empty_upload() -> DatasetBundle:
        return DatasetBundle(
            raw_df=pd.DataFrame(),
            daily_df=pd.DataFrame(),
            processed_df=pd.DataFrame(),
            source_label="Uploaded CSV",
            source_detail="No file uploaded yet",
            split_strategy="chronological_ratio",
        )

    @staticmethod
    def _process_csv(path_or_buffer: str | Path | Any, source_label: str, source_detail: str, split_strategy: str) -> DatasetBundle:
        raw_df = DatasetService.read_csv(path_or_buffer)
        daily_df = daily_ohlcv(raw_df)
        if len(daily_df) < MODEL_WINDOW_SIZE:
            raise CandleDataError(
                f"Need at least {MODEL_WINDOW_SIZE} daily candles after aggregation; "
                f"found {len(daily_df)}"
            )
        if split_strategy == "paper_date":
            processed_df = add_dataset_split(daily_df)
        elif split_strategy == "chronological_ratio":
            processed_df = add_chronological_split(daily_df)
        else:
            raise ValueError(f"Unsupported split strategy: {split_strategy}")
        return DatasetBundle(
            raw_df=raw_df,
            daily_df=daily_df,
            processed_df=processed_df,
            source_label=source_label,
            source_detail=source_detail,
            split_strategy=split_strategy,
        )

    @staticmethod
    def read_csv(path_or_buffer: str | Path | Any) -> pd.DataFrame:
        DatasetService._rewind(path_or_buffer)
        frame = pd.read_csv(path_or_buffer)
        if len(frame.columns) > 1:
            return frame

        DatasetService._rewind(path_or_buffer)
        return pd.read_csv(path_or_buffer, sep=None, engine="python")

    @staticmethod
    def _rewind(path_or_buffer: str | Path | Any) -> None:
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)
