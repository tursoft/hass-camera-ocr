"""Constants for the Camera Data Extractor integration."""

DOMAIN = "hass_camera_ocr"

# Configuration keys
CONF_CAMERA_NAME = "camera_name"
CONF_STREAM_URL = "stream_url"
CONF_USERNAME = "username"
CONF_PASSWORD = "password"
CONF_SCAN_INTERVAL = "scan_interval"
CONF_ROI_X = "roi_x"
CONF_ROI_Y = "roi_y"
CONF_ROI_WIDTH = "roi_width"
CONF_ROI_HEIGHT = "roi_height"
CONF_VALUE_NAME = "value_name"
CONF_UNIT_OF_MEASUREMENT = "unit_of_measurement"
CONF_DECIMAL_PLACES = "decimal_places"
CONF_PREPROCESSING = "preprocessing"

# Default values
DEFAULT_SCAN_INTERVAL = 30
DEFAULT_ROI_X = 0
DEFAULT_ROI_Y = 0
DEFAULT_ROI_WIDTH = 0
DEFAULT_ROI_HEIGHT = 0
DEFAULT_DECIMAL_PLACES = 1
DEFAULT_PREPROCESSING = "auto"

# Preprocessing options
PREPROCESSING_NONE = "none"
PREPROCESSING_AUTO = "auto"
PREPROCESSING_THRESHOLD = "threshold"
PREPROCESSING_ADAPTIVE = "adaptive"
PREPROCESSING_INVERT = "invert"

PREPROCESSING_OPTIONS = [
    PREPROCESSING_NONE,
    PREPROCESSING_AUTO,
    PREPROCESSING_THRESHOLD,
    PREPROCESSING_ADAPTIVE,
    PREPROCESSING_INVERT,
]

# Service names
SERVICE_CAPTURE_FRAME = "capture_frame"
SERVICE_UPDATE_ROI = "update_roi"

# Attributes
ATTR_RAW_TEXT = "raw_text"
ATTR_CONFIDENCE = "confidence"
ATTR_LAST_FRAME_TIME = "last_frame_time"
ATTR_CAMERA_NAME = "camera_name"
