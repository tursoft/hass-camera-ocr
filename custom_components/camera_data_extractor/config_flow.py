"""Config flow for Camera Data Extractor integration."""
from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_PASSWORD, CONF_USERNAME
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    DOMAIN,
    CONF_CAMERA_NAME,
    CONF_STREAM_URL,
    CONF_SCAN_INTERVAL,
    CONF_ROI_X,
    CONF_ROI_Y,
    CONF_ROI_WIDTH,
    CONF_ROI_HEIGHT,
    CONF_VALUE_NAME,
    CONF_UNIT_OF_MEASUREMENT,
    CONF_DECIMAL_PLACES,
    CONF_PREPROCESSING,
    DEFAULT_SCAN_INTERVAL,
    DEFAULT_ROI_X,
    DEFAULT_ROI_Y,
    DEFAULT_ROI_WIDTH,
    DEFAULT_ROI_HEIGHT,
    DEFAULT_DECIMAL_PLACES,
    DEFAULT_PREPROCESSING,
    PREPROCESSING_OPTIONS,
)

_LOGGER = logging.getLogger(__name__)


class CameraDataExtractorConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Camera Data Extractor."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._data: dict[str, Any] = {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step - camera connection details."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validate the stream URL by attempting to connect
            stream_url = user_input[CONF_STREAM_URL]
            username = user_input.get(CONF_USERNAME, "")
            password = user_input.get(CONF_PASSWORD, "")

            # Build authenticated URL if credentials provided
            if username and password:
                # Insert credentials into URL
                if "://" in stream_url:
                    protocol, rest = stream_url.split("://", 1)
                    stream_url = f"{protocol}://{username}:{password}@{rest}"

            # Test connection
            valid = await self._test_stream_connection(stream_url)
            if valid:
                self._data = user_input
                return await self.async_step_extraction()
            else:
                errors["base"] = "cannot_connect"

        data_schema = vol.Schema(
            {
                vol.Required(CONF_CAMERA_NAME): str,
                vol.Required(CONF_STREAM_URL): str,
                vol.Optional(CONF_USERNAME, default=""): str,
                vol.Optional(CONF_PASSWORD, default=""): str,
            }
        )

        return self.async_show_form(
            step_id="user",
            data_schema=data_schema,
            errors=errors,
            description_placeholders={
                "stream_url_example": "rtsp://192.168.1.100:554/stream1 or http://192.168.1.100/video"
            },
        )

    async def async_step_extraction(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the extraction configuration step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_roi()

        data_schema = vol.Schema(
            {
                vol.Required(CONF_VALUE_NAME, default="Temperature"): str,
                vol.Optional(CONF_UNIT_OF_MEASUREMENT, default="°C"): str,
                vol.Optional(
                    CONF_DECIMAL_PLACES, default=DEFAULT_DECIMAL_PLACES
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, max=5, step=1, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Optional(
                    CONF_SCAN_INTERVAL, default=DEFAULT_SCAN_INTERVAL
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=5, max=3600, step=1, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Optional(
                    CONF_PREPROCESSING, default=DEFAULT_PREPROCESSING
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=PREPROCESSING_OPTIONS,
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
            }
        )

        return self.async_show_form(
            step_id="extraction",
            data_schema=data_schema,
            errors=errors,
        )

    async def async_step_roi(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the ROI (Region of Interest) configuration step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._data.update(user_input)

            # Create unique ID based on camera name
            await self.async_set_unique_id(
                f"{DOMAIN}_{self._data[CONF_CAMERA_NAME].lower().replace(' ', '_')}"
            )
            self._abort_if_unique_id_configured()

            return self.async_create_entry(
                title=f"{self._data[CONF_CAMERA_NAME]} - {self._data[CONF_VALUE_NAME]}",
                data=self._data,
            )

        data_schema = vol.Schema(
            {
                vol.Optional(CONF_ROI_X, default=DEFAULT_ROI_X): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, max=4096, step=1, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Optional(CONF_ROI_Y, default=DEFAULT_ROI_Y): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, max=4096, step=1, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Optional(
                    CONF_ROI_WIDTH, default=DEFAULT_ROI_WIDTH
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, max=4096, step=1, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Optional(
                    CONF_ROI_HEIGHT, default=DEFAULT_ROI_HEIGHT
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, max=4096, step=1, mode=selector.NumberSelectorMode.BOX
                    )
                ),
            }
        )

        return self.async_show_form(
            step_id="roi",
            data_schema=data_schema,
            errors=errors,
            description_placeholders={
                "roi_hint": "Set all to 0 to use the full frame. Otherwise, specify the region containing the value to extract."
            },
        )

    async def _test_stream_connection(self, stream_url: str) -> bool:
        """Test if we can connect to the video stream."""
        try:
            import cv2

            def test_connection():
                cap = cv2.VideoCapture(stream_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                success = cap.isOpened()
                if success:
                    ret, _ = cap.read()
                    success = ret
                cap.release()
                return success

            # Run in executor to not block
            result = await self.hass.async_add_executor_job(test_connection)
            return result
        except Exception as ex:
            _LOGGER.error("Error testing stream connection: %s", ex)
            return False

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create the options flow."""
        return CameraDataExtractorOptionsFlow(config_entry)


class CameraDataExtractorOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for Camera Data Extractor."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        current = self.config_entry.data

        data_schema = vol.Schema(
            {
                vol.Optional(
                    CONF_SCAN_INTERVAL,
                    default=current.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=5, max=3600, step=1, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Optional(
                    CONF_ROI_X, default=current.get(CONF_ROI_X, DEFAULT_ROI_X)
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, max=4096, step=1, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Optional(
                    CONF_ROI_Y, default=current.get(CONF_ROI_Y, DEFAULT_ROI_Y)
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, max=4096, step=1, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Optional(
                    CONF_ROI_WIDTH,
                    default=current.get(CONF_ROI_WIDTH, DEFAULT_ROI_WIDTH),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, max=4096, step=1, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Optional(
                    CONF_ROI_HEIGHT,
                    default=current.get(CONF_ROI_HEIGHT, DEFAULT_ROI_HEIGHT),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, max=4096, step=1, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Optional(
                    CONF_PREPROCESSING,
                    default=current.get(CONF_PREPROCESSING, DEFAULT_PREPROCESSING),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=PREPROCESSING_OPTIONS,
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(
                    CONF_DECIMAL_PLACES,
                    default=current.get(CONF_DECIMAL_PLACES, DEFAULT_DECIMAL_PLACES),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, max=5, step=1, mode=selector.NumberSelectorMode.BOX
                    )
                ),
            }
        )

        return self.async_show_form(step_id="init", data_schema=data_schema)
