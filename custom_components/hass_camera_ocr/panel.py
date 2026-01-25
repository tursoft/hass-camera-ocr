"""Panel registration for Camera Data Extractor web UI."""
from __future__ import annotations

import logging
from pathlib import Path

from homeassistant.components import frontend
from homeassistant.components.http.static import StaticPathConfig
from homeassistant.core import HomeAssistant

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

PANEL_URL = "/camera-data-extractor"
PANEL_TITLE = "Camera Data Extractor"
PANEL_ICON = "mdi:camera-enhance"


async def async_register_panel(hass: HomeAssistant) -> None:
    """Register the Camera Data Extractor panel."""
    # Get the path to our www folder
    panel_path = Path(__file__).parent / "www"

    # Register static path for our panel files
    await hass.http.async_register_static_paths([
        StaticPathConfig(
            "/hass_camera_ocr/static",
            str(panel_path),
            cache_headers=False,
        )
    ])

    # Register the panel
    frontend.async_register_built_in_panel(
        hass,
        component_name="iframe",
        sidebar_title=PANEL_TITLE,
        sidebar_icon=PANEL_ICON,
        frontend_url_path="camera-data-extractor",
        config={
            "url": "/hass_camera_ocr/static/panel.html",
        },
        require_admin=False,
    )

    _LOGGER.info("Registered Camera Data Extractor panel")


async def async_unregister_panel(hass: HomeAssistant) -> None:
    """Unregister the Camera Data Extractor panel."""
    try:
        frontend.async_remove_panel(hass, "camera-data-extractor")
        _LOGGER.info("Unregistered Camera Data Extractor panel")
    except Exception:
        pass
