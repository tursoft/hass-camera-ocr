# Changelog

## [1.2.3] - 2026

### Fixed
- Version bump to trigger add-on update

## [1.2.2] - 2026

### Fixed
- Fixed RTSP stream capture by adding FFmpeg support
- Use TCP transport for more reliable RTSP connections
- Added connection timeouts to prevent hanging
- Improved error messages for stream connection failures
- Added logging for stream capture debugging

## [1.2.1] - 2026

### Added
- URL Builder in Add Camera dialog - toggle between full URL input or build from components
- Bidirectional URL parsing - enter full URL to auto-populate fields, or fill fields to generate URL
- Support for RTSP, HTTP, HTTPS protocols with host, port, path, username, password fields
- Live URL preview when using Build URL mode
- Common stream path hints for easier configuration

## [1.2.0] - 2026

### Changed
- Renamed project from "Camera Data Extractor" to "Camera OCR"
- Renamed repository from ha-camera-data-extractor to hass-camera-ocr
- Updated all internal references to new naming

## [1.1.2] - 2026

### Added
- Persistent storage in /config directory - camera config survives uninstall/reinstall
- Templates stored in /config/hass_camera_ocr/templates/ for persistence
- Auto-migration from options.json to persistent storage on first run

### Fixed
- Camera configuration now persists across add-on updates and reinstalls

## [1.1.1] - 2026

### Fixed
- Fix API URLs to use relative paths for Home Assistant ingress compatibility

## [1.1.0] - 2026

### Added
- Full admin web interface with Dashboard, Cameras, Live Preview, Templates, and Discovery pages
- Visual ROI selection - click and drag on camera preview to select extraction area
- Live preview with ROI borders and extracted value overlay
- Template matching for automatic ROI tracking (works even when camera rotates/moves)
- ONVIF camera discovery to automatically find cameras on network
- Template saving and management
- Camera CRUD operations from web UI (add, edit, delete)
- Real-time value display with confidence indicators
- Toast notifications for user feedback
- Dark theme matching Home Assistant style

### Changed
- Complete rewrite of web UI with modern, user-friendly design
- Improved preprocessing options

## [1.0.1] - 2026

### Fixed
- Fix OpenCV GStreamer binding error on Alpine Linux
- Fix s6-test not found error in service finish script
- Use pip opencv-python-headless for better compatibility

## [1.0.0] - 2026

### Added
- Initial release
- Support for RTSP and HTTP camera streams
- OCR-based numeric value extraction
- Region of Interest (ROI) configuration
- Multiple preprocessing modes (auto, threshold, adaptive, invert)
- Web UI for monitoring extracted values
- Home Assistant Ingress support
