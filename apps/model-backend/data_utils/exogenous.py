"""Track 3 rung 1 — exogenous daily features from the frozen snapshot.

Registered protocol: .ai/phase-track3-exogenous.md §2-3 (workspace root).

The snapshot under data/exogenous/ stores RAW values (see manifest.json); this
module owns everything that turns them into model-ready dataframe columns:
  - static transforms: fng/100, log1p for counts/flows/hashrate, MVRV raw;
  - the conservative t-1 shift: the value usable on day t is the metric
    published for day t-1, so publication timing can never leak;
  - the calendar-day join into the split dataframes. Split Timestamps are
    local-midnight epochs of whatever machine generated the data cache (UTC-7
    in the committed files), so the join key is recovered with a +12h rule
    that is exact for any UTC offset in (-12, +12).
In-window normalization of these channels happens later, in DataTransform.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

SNAPSHOT_DIR = Path(__file__).resolve().parents[1] / 'data' / 'exogenous'


def _div100(series: pd.Series) -> pd.Series:
    return series.astype(float) / 100.0


def _log1p(series: pd.Series) -> pd.Series:
    return np.log1p(series.astype(float))


def _raw(series: pd.Series) -> pd.Series:
    return series.astype(float)


# channel name -> (snapshot file, source column, static transform)
EXOGENOUS_FEATURES = {
    'fng':         ('fng.csv', 'fng', _div100),
    'adr_act':     ('coinmetrics_btc.csv', 'AdrActCnt', _log1p),
    'tx_cnt':      ('coinmetrics_btc.csv', 'TxCnt', _log1p),
    'mvrv':        ('coinmetrics_btc.csv', 'CapMVRVCur', _raw),
    'flow_in_ex':  ('coinmetrics_btc.csv', 'FlowInExUSD', _log1p),
    'flow_out_ex': ('coinmetrics_btc.csv', 'FlowOutExUSD', _log1p),
    'hash_rate':   ('coinmetrics_btc.csv', 'HashRate', _log1p),
}


def validate_features(features: Iterable[str]) -> None:
    unknown = set(features) - set(EXOGENOUS_FEATURES)
    if unknown:
        raise ValueError(f'Unknown exogenous features: {sorted(unknown)}')


def timestamp_to_day(ts: int) -> date:
    """Calendar day of a local-midnight epoch, exact for UTC offsets in (-12, +12)."""
    return datetime.fromtimestamp(int(ts) + 43200, tz=timezone.utc).date()


def load_exogenous_frame(features: List[str],
                         snapshot_dir: Path = SNAPSHOT_DIR) -> pd.DataFrame:
    """Frozen snapshot -> one row per day with transformed, t-1-shifted values."""
    validate_features(features)
    frame = None
    for file_name in sorted({EXOGENOUS_FEATURES[f][0] for f in features}):
        raw = pd.read_csv(snapshot_dir / file_name)
        part = {'day': [date.fromisoformat(d) + timedelta(days=1)  # t-1 shift
                        for d in raw['date']]}
        for name in features:
            source_file, column, transform = EXOGENOUS_FEATURES[name]
            if source_file == file_name:
                part[name] = transform(raw[column])
        part = pd.DataFrame(part)
        frame = part if frame is None else frame.merge(part, on='day', how='inner')
    return frame[['day'] + list(features)]


def merge_exogenous(split_df: pd.DataFrame, exo_df: pd.DataFrame,
                    features: List[str]) -> pd.DataFrame:
    """Left-join exogenous columns onto a split dataframe by calendar day.

    Raises if any split day lacks coverage — a silent gap would train on NaN.
    """
    validate_features(features)
    out = split_df.copy()
    out['_exo_day'] = out['Timestamp'].map(timestamp_to_day)
    merged = out.merge(exo_df[['day'] + list(features)], left_on='_exo_day',
                       right_on='day', how='left', validate='many_to_one')
    gaps = merged.loc[merged[list(features)].isna().any(axis=1), '_exo_day']
    if len(gaps):
        raise ValueError(
            f'Exogenous coverage missing for {len(gaps)} day(s), '
            f'e.g. {sorted(set(gaps))[:5]} — extend the frozen snapshot.')
    return merged.drop(columns=['_exo_day', 'day'])
