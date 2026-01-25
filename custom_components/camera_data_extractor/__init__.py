"""The Camera Data Extractor integration."""
from __future__ import annotations

import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
import voluptuous as vol

from .const import (
    DOMAIN,
    CONF_ROI_X,
    CONF_ROI_Y,
    CONF_ROI_WIDTH,
    CONF_ROI_HEIGHT,
    SERVICE_CAPTURE_FRAME,
    SERVICE_UPDATE_ROI,
)

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.SENSOR]

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

# Track if API and panel are registered
_API_REGISTERED = False
_PANEL_REGISTERED = False


async def async_setup(hass: HomeAssistant, config: dict[str, Any]) -> bool:
    """Set up the Camera Data Extractor component."""
    hass.data.setdefault(DOMAIN, {})

    # Register API endpoints once
    global _API_REGISTERED
    if not _API_REGISTERED:
        from .api import async_register_api
        async_register_api(hass)
        _API_REGISTERED = True
        _LOGGER.info("Registered Camera Data Extractor API endpoints")

    # Register discovery API
    from .discovery import discover_cameras, CameraDiscovery

    async def handle_discover(request):
        """Handle camera discovery API request."""
        from aiohttp import web
        try:
            discovery = CameraDiscovery()
            cameras = await discovery.discover(timeout=10.0)
            return web.json_response({
                "success": True,
                "cameras": [
                    {
                        "ip": cam.ip,
                        "port": cam.port,
                        "name": cam.name,
                        "manufacturer": cam.manufacturer,
                        "model": cam.model,
                        "streams": cam.streams[:5] if cam.streams else [],
                    }
                    for cam in cameras
                ],
            })
        except Exception as ex:
            _LOGGER.error("Discovery error: %s", ex)
            return web.json_response({"error": str(ex)}, status=500)

    hass.http.register_view(DiscoveryView(hass))

    # Register panel
    global _PANEL_REGISTERED
    if not _PANEL_REGISTERED:
        try:
            from .panel import async_register_panel
            await async_register_panel(hass)
            _PANEL_REGISTERED = True
        except Exception as ex:
            _LOGGER.warning("Could not register panel: %s", ex)

    return True


class DiscoveryView:
    """Discovery API view."""

    url = "/api/camera_data_extractor/discover"
    name = "api:camera_data_extractor:discover"
    requires_auth = True

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize."""
        self.hass = hass
        # Required for Home Assistant HTTP registration
        self.extra_urls = []
        self.cors_allowed = False

    async def get(self, request):
        """Handle GET request for camera discovery."""
        from aiohttp import web
        from .discovery import CameraDiscovery

        try:
            discovery = CameraDiscovery()
            cameras = await discovery.discover(timeout=10.0)
            return web.json_response({
                "success": True,
                "cameras": [
                    {
                        "ip": cam.ip,
                        "port": cam.port,
                        "name": cam.name,
                        "manufacturer": cam.manufacturer,
                        "model": cam.model,
                        "streams": cam.streams[:5] if cam.streams else [],
                    }
                    for cam in cameras
                ],
            })
        except Exception as ex:
            _LOGGER.error("Discovery error: %s", ex)
            return web.json_response({"error": str(ex)}, status=500)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Camera Data Extractor from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = dict(entry.data)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register services (only once)
    async def handle_capture_frame(call: ServiceCall) -> None:
        """Handle the capture_frame service call."""
        entity_id = call.data.get("entity_id")
        if entity_id:
            await hass.services.async_call(
                "homeassistant",
                "update_entity",
                {"entity_id": entity_id},
                blocking=True,
            )

    async def handle_update_roi(call: ServiceCall) -> None:
        """Handle the update_roi service call."""
        entity_id = call.data.get("entity_id")
        roi_x = call.data.get(CONF_ROI_X)
        roi_y = call.data.get(CONF_ROI_Y)
        roi_width = call.data.get(CONF_ROI_WIDTH)
        roi_height = call.data.get(CONF_ROI_HEIGHT)

        for eid, data in hass.data[DOMAIN].items():
            if isinstance(data, dict) and data.get("entity_id") == entity_id:
                new_data = dict(data)
                if roi_x is not None:
                    new_data[CONF_ROI_X] = roi_x
                if roi_y is not None:
                    new_data[CONF_ROI_Y] = roi_y
                if roi_width is not None:
                    new_data[CONF_ROI_WIDTH] = roi_width
                if roi_height is not None:
                    new_data[CONF_ROI_HEIGHT] = roi_height
                hass.data[DOMAIN][eid] = new_data
                _LOGGER.info("Updated ROI for %s", entity_id)
                break

    async def handle_use_template(call: ServiceCall) -> None:
        """Handle the use_template service call."""
        entity_id = call.data.get("entity_id")
        template_name = call.data.get("template_name")

        from .template_matcher import TemplateMatcher
        from .camera_processor import CameraProcessor

        # Find the entry for this entity
        for eid, data in hass.data[DOMAIN].items():
            if isinstance(data, dict) and data.get("entity_id") == entity_id:
                # Get stream info
                stream_url = data.get("stream_url")
                username = data.get("username")
                password = data.get("password")

                if not stream_url:
                    _LOGGER.error("No stream URL for entity %s", entity_id)
                    return

                # Capture frame
                processor = CameraProcessor(stream_url, username, password)
                frame_result = await hass.async_add_executor_job(
                    processor.capture_frame
                )

                if not frame_result.success:
                    _LOGGER.error("Failed to capture frame for template matching")
                    return

                # Find region using template
                matcher = hass.data[DOMAIN].get("matcher")
                if not matcher:
                    matcher = TemplateMatcher()
                    hass.data[DOMAIN]["matcher"] = matcher

                match_result = await hass.async_add_executor_job(
                    matcher.find_region,
                    frame_result.frame,
                    template_name,
                )

                if match_result.found:
                    # Update ROI with matched region
                    data[CONF_ROI_X] = match_result.x
                    data[CONF_ROI_Y] = match_result.y
                    data[CONF_ROI_WIDTH] = match_result.width
                    data[CONF_ROI_HEIGHT] = match_result.height
                    hass.data[DOMAIN][eid] = data
                    _LOGGER.info(
                        "Updated ROI from template match: (%d, %d, %d, %d) confidence: %.1f%%",
                        match_result.x,
                        match_result.y,
                        match_result.width,
                        match_result.height,
                        match_result.confidence,
                    )
                else:
                    _LOGGER.warning(
                        "Template '%s' not found in current frame: %s",
                        template_name,
                        match_result.error,
                    )
                break

    if not hass.services.has_service(DOMAIN, SERVICE_CAPTURE_FRAME):
        hass.services.async_register(
            DOMAIN,
            SERVICE_CAPTURE_FRAME,
            handle_capture_frame,
            schema=vol.Schema({vol.Required("entity_id"): cv.entity_id}),
        )

    if not hass.services.has_service(DOMAIN, SERVICE_UPDATE_ROI):
        hass.services.async_register(
            DOMAIN,
            SERVICE_UPDATE_ROI,
            handle_update_roi,
            schema=vol.Schema(
                {
                    vol.Required("entity_id"): cv.entity_id,
                    vol.Optional(CONF_ROI_X): cv.positive_int,
                    vol.Optional(CONF_ROI_Y): cv.positive_int,
                    vol.Optional(CONF_ROI_WIDTH): cv.positive_int,
                    vol.Optional(CONF_ROI_HEIGHT): cv.positive_int,
                }
            ),
        )

    if not hass.services.has_service(DOMAIN, "use_template"):
        hass.services.async_register(
            DOMAIN,
            "use_template",
            handle_use_template,
            schema=vol.Schema(
                {
                    vol.Required("entity_id"): cv.entity_id,
                    vol.Required("template_name"): cv.string,
                }
            ),
        )

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id, None)

    return unload_ok


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)
