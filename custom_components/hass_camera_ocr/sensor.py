"""Sensor platform for Camera Data Extractor."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_PASSWORD, CONF_USERNAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.util import slugify

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
    ATTR_RAW_TEXT,
    ATTR_CONFIDENCE,
    ATTR_LAST_FRAME_TIME,
    ATTR_CAMERA_NAME,
)
from .camera_processor import CameraProcessor
from .ocr_engine import OCREngine

_LOGGER = logging.getLogger(__name__)

SCAN_INTERVAL = timedelta(seconds=30)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the sensor platform."""
    config = entry.data

    # Create the sensor entity
    sensor = CameraDataExtractorSensor(
        hass=hass,
        entry_id=entry.entry_id,
        camera_name=config[CONF_CAMERA_NAME],
        stream_url=config[CONF_STREAM_URL],
        username=config.get(CONF_USERNAME),
        password=config.get(CONF_PASSWORD),
        value_name=config.get(CONF_VALUE_NAME, "Value"),
        unit_of_measurement=config.get(CONF_UNIT_OF_MEASUREMENT),
        scan_interval=config.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL),
        roi_x=config.get(CONF_ROI_X, DEFAULT_ROI_X),
        roi_y=config.get(CONF_ROI_Y, DEFAULT_ROI_Y),
        roi_width=config.get(CONF_ROI_WIDTH, DEFAULT_ROI_WIDTH),
        roi_height=config.get(CONF_ROI_HEIGHT, DEFAULT_ROI_HEIGHT),
        decimal_places=config.get(CONF_DECIMAL_PLACES, DEFAULT_DECIMAL_PLACES),
        preprocessing=config.get(CONF_PREPROCESSING, DEFAULT_PREPROCESSING),
    )

    async_add_entities([sensor], True)


class CameraDataExtractorSensor(SensorEntity):
    """Sensor entity for extracting data from camera streams."""

    _attr_has_entity_name = True
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        camera_name: str,
        stream_url: str,
        username: str | None,
        password: str | None,
        value_name: str,
        unit_of_measurement: str | None,
        scan_interval: int,
        roi_x: int,
        roi_y: int,
        roi_width: int,
        roi_height: int,
        decimal_places: int,
        preprocessing: str,
    ) -> None:
        """Initialize the sensor."""
        self.hass = hass
        self._entry_id = entry_id
        self._camera_name = camera_name
        self._stream_url = stream_url
        self._username = username
        self._password = password
        self._value_name = value_name
        self._scan_interval = scan_interval
        self._roi = (roi_x, roi_y, roi_width, roi_height)
        self._decimal_places = decimal_places
        self._preprocessing = preprocessing

        # Entity attributes
        self._attr_name = value_name
        self._attr_native_unit_of_measurement = unit_of_measurement
        self._attr_unique_id = f"{DOMAIN}_{slugify(camera_name)}_{slugify(value_name)}"

        # Determine device class based on unit
        if unit_of_measurement in ("°C", "°F", "K"):
            self._attr_device_class = SensorDeviceClass.TEMPERATURE
        elif unit_of_measurement in ("%",):
            self._attr_device_class = SensorDeviceClass.HUMIDITY
        elif unit_of_measurement in ("bar", "psi", "Pa", "hPa", "mbar"):
            self._attr_device_class = SensorDeviceClass.PRESSURE
        else:
            self._attr_device_class = None

        # State
        self._attr_native_value: float | None = None
        self._raw_text: str = ""
        self._confidence: float = 0
        self._last_frame_time: datetime | None = None
        self._available = True

        # Processors
        self._camera_processor: CameraProcessor | None = None
        self._ocr_engine: OCREngine | None = None

    @property
    def device_info(self) -> DeviceInfo:
        """Return device information."""
        return DeviceInfo(
            identifiers={(DOMAIN, f"{self._camera_name}")},
            name=f"Camera: {self._camera_name}",
            manufacturer="Camera Data Extractor",
            model="OCR Sensor",
            sw_version="1.0.0",
        )

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional state attributes."""
        return {
            ATTR_RAW_TEXT: self._raw_text,
            ATTR_CONFIDENCE: self._confidence,
            ATTR_LAST_FRAME_TIME: self._last_frame_time.isoformat()
            if self._last_frame_time
            else None,
            ATTR_CAMERA_NAME: self._camera_name,
            "roi_x": self._roi[0],
            "roi_y": self._roi[1],
            "roi_width": self._roi[2],
            "roi_height": self._roi[3],
            "preprocessing": self._preprocessing,
            "stream_url": self._stream_url,
        }

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return self._available

    async def async_added_to_hass(self) -> None:
        """Run when entity is added to hass."""
        # Initialize processors
        self._camera_processor = CameraProcessor(
            self._stream_url, self._username, self._password
        )
        self._ocr_engine = OCREngine(decimal_places=self._decimal_places)

        # Store entity_id in hass data for service calls
        self.hass.data[DOMAIN][self._entry_id] = {
            **self.hass.data[DOMAIN].get(self._entry_id, {}),
            "entity_id": self.entity_id,
        }

    async def async_will_remove_from_hass(self) -> None:
        """Run when entity is being removed."""
        self._camera_processor = None
        self._ocr_engine = None

    async def async_update(self) -> None:
        """Update the sensor value."""
        if not self._camera_processor or not self._ocr_engine:
            return

        # Check if ROI was updated via service
        entry_data = self.hass.data[DOMAIN].get(self._entry_id, {})
        if isinstance(entry_data, dict):
            new_roi = (
                entry_data.get(CONF_ROI_X, self._roi[0]),
                entry_data.get(CONF_ROI_Y, self._roi[1]),
                entry_data.get(CONF_ROI_WIDTH, self._roi[2]),
                entry_data.get(CONF_ROI_HEIGHT, self._roi[3]),
            )
            if new_roi != self._roi:
                self._roi = new_roi
                _LOGGER.debug("ROI updated to %s", self._roi)

        try:
            # Capture frame in executor
            frame_result = await self.hass.async_add_executor_job(
                self._camera_processor.capture_frame
            )

            if not frame_result.success:
                _LOGGER.warning(
                    "Failed to capture frame from %s: %s",
                    self._camera_name,
                    frame_result.error,
                )
                self._available = False
                return

            self._available = True
            self._last_frame_time = datetime.fromtimestamp(frame_result.timestamp)

            # Extract value using OCR
            ocr_result = await self.hass.async_add_executor_job(
                self._ocr_engine.extract_value,
                frame_result.frame,
                self._preprocessing,
                self._roi if any(self._roi) else None,
            )

            self._raw_text = ocr_result.raw_text
            self._confidence = ocr_result.confidence

            if ocr_result.success and ocr_result.value is not None:
                self._attr_native_value = ocr_result.value
                _LOGGER.debug(
                    "Extracted value %s from %s (confidence: %.1f%%)",
                    ocr_result.value,
                    self._camera_name,
                    ocr_result.confidence,
                )
            else:
                _LOGGER.debug(
                    "OCR extraction failed for %s: %s (raw text: '%s')",
                    self._camera_name,
                    ocr_result.error,
                    ocr_result.raw_text,
                )
                # Keep the last valid value

        except Exception as ex:
            _LOGGER.error("Error updating %s: %s", self._camera_name, ex)
            self._available = False
