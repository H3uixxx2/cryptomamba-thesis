"""Import facade for the reused CryptoMamba logic.

This is the ONLY place the console reaches into the Streamlit repo's package.
Everything is re-exported so routers depend on ``server.logic`` rather than on the
external path, and so a missing/moved source repo fails loudly here with a clear
message instead of as an obscure ImportError deep in a request handler.
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
