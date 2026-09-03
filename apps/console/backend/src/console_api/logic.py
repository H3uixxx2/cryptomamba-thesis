"""Single import point for the vendored ``cryptomamba_ui`` modules.

Services import from here rather than from ``.vendor.cryptomamba_ui`` directly, so
the vendored package can be re-synced or replaced by touching one file. A broken
vendor tree raises here at import time with a readable message instead of as an
ImportError deep inside a request handler.
"""
from __future__ import annotations

from .core import config


try:  # reused, unchanged logic
    from console_api.vendor.cryptomamba_ui import data as data  # noqa: F401
    from console_api.vendor.cryptomamba_ui import charts as charts  # noqa: F401
    from console_api.vendor.cryptomamba_ui import trading_logic as trading_logic  # noqa: F401
    from console_api.vendor.cryptomamba_ui.data import CandleDataError  # noqa: F401
    from console_api.vendor.cryptomamba_ui.dataset_service import (  # noqa: F401
        DatasetBundle,
        DatasetService,
    )
    from console_api.vendor.cryptomamba_ui.api_client import (  # noqa: F401
        ApiClientError,
        CryptoMambaApiClient,
    )
    from console_api.vendor.cryptomamba_ui.predict_artifacts import (  # noqa: F401
        OfflinePredictionError,
        load_offline_prediction,
    )
except ImportError as exc:  # pragma: no cover - environment wiring failure
    raise ImportError(
        "Could not import the vendored cryptomamba_ui package "
        "(console_api.vendor.cryptomamba_ui). "
        f"Original error: {exc}"
    ) from exc


# Shared dataset service (paper sample fixture + display root for relative paths).
DATASET_SERVICE = DatasetService(
    sample_path=config.SAMPLE_DATA_PATH,
    display_root=config.CONSOLE_ROOT,
)
