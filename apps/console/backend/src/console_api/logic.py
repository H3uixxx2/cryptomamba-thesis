"""Single import point for the vendored ``cryptomamba_ui`` modules; a broken vendor tree fails here at import."""
from __future__ import annotations

from .core import config


try:
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
except ImportError as exc:  # pragma: no cover - environment wiring failure
    raise ImportError(
        "Could not import the vendored cryptomamba_ui package "
        "(console_api.vendor.cryptomamba_ui). "
        f"Original error: {exc}"
    ) from exc


# Shared dataset service: the paper sample fixture, and the root that relative paths are shown against.
DATASET_SERVICE = DatasetService(
    sample_path=config.SAMPLE_DATA_PATH,
    display_root=config.CONSOLE_ROOT,
)
