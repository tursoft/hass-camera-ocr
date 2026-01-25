# Changelog

## [1.2.16] - 2026

### Changed
- Version bump to trigger add-on update

## [1.2.15] - 2026

### Added
- **ROI Thumbnails in Value History**: Shows cropped ROI images for each extraction
- **History Detail Dialog**: Click on any history entry to view full details (image, value, date, OCR provider, confidence, video description)
- **Value History View Modes**: Switch between Table, Card, and Chart views
- **Value Range Filtering**: Set min/max expected values to ignore bad OCR readings
- **AI Provider Test with Upload**: Upload images or use camera preview to test OCR providers
- Camera icon thumbnails in Network Discovery results
- Direct ONVIF device probing for cameras that don't respond to WS-Discovery
- ONVIF port information shown in discovery results
- OCR provider tracking for each extraction (shows Tesseract or AI provider used)

### Improved
- Network Discovery with HTTP-based camera detection (finds cameras like Tapo)
- Port scanner now detects camera-specific ports (2020 for Tapo, 1935 for RTMP)
- Better camera placeholder display when credentials are required
- Password masking in camera URLs on the Cameras page
- Default unit now set to °C in Add Camera form

### Fixed
- AI test button now properly handles capture frame tuple return
- ONVIF WS-Discovery now sends multiple probes for better reliability
- Discovery now binds to local network interface

## [1.2.14] - 2026

### Added
- **Home Assistant Entity Integration**: Automatically exposes sensor entities for each camera
  - `sensor.camera_ocr_{name}_value` - Numeric value
  - `sensor.camera_ocr_{name}_text` - Raw OCR text
  - `sensor.camera_ocr_{name}_confidence` - Confidence percentage
- Sensors can be used in dashboards and automations

### Documentation
- Added Home Assistant entity documentation with automation examples

## [1.2.13] - 2026

### Added
- Order number column (#) in value history table
- Pause/Resume buttons for live value updates on dashboard
- Edit camera button in Live Preview section
- Google Document AI as cloud OCR provider

### Improved
- PTZ controls now try multiple profile tokens and endpoints for better compatibility

## [1.2.12] - 2026

### Added
- Cloud OCR service providers support:
  - Google Cloud Vision API
  - Azure Computer Vision (Read API)
  - AWS Textract
- Custom OpenAI-compatible API endpoint support
- Low confidence visual indicators (red text when confidence < 80%)
- Confidence progress bar in value history table
- Right-aligned value column in history table

### Changed
- Improved AI provider configuration UI with better organization
- Updated model hints for all AI providers

## [1.2.11] - 2026

### Added
- Test button on each saved ROI to re-run extraction
- Test All ROIs button to find best configuration
- Saved ROIs now outside scrollable area (not affected by zoom/scroll)

### Improved
- OCR accuracy with better preprocessing:
  - Added padding to prevent digit cutoff
  - Larger scaling for small images (5x for tiny regions)
  - Morphological cleanup for digit segments
  - Sharpening filter to enhance digit edges
  - Better adaptive thresholding

### Changed
- Reorganized Live Preview layout for better usability

## [1.2.10] - 2026

### Added
- Persistent value history - survives uninstall/reinstall
- Saved ROI thumbnails below live preview with extracted values
- Save/delete/apply saved ROIs
- Loading indicator for Test Extract button

### Fixed
- Value history now persists to disk in /config directory

## [1.2.9] - 2026

### Changed
- Version bump to trigger add-on update

## [1.2.8] - 2026

### Added
- PTZ (Pan-Tilt) controls in Live Preview for ONVIF cameras
- PTZ buttons: Up, Down, Left, Right, Home position
- ROI preview in Test Extract mode - shows cropped region being analyzed
- AI Integration for enhanced OCR and scene description
  - Support for OpenAI (GPT-4o), Anthropic (Claude), Google (Gemini), and Ollama (local)
  - AI-powered OCR enhancement for better value extraction
  - Video scene description generation (exposed as `video_description` attribute)
  - New AI Settings page for configuration
- Video description display on dashboard for each camera

### Changed
- Replaced rotate buttons with PTZ controls
- Simplified zoom interface

## [1.2.7] - 2026

### Added
- Camera preview thumbnails in discovery results
- Auto-load previews when cameras are discovered
- Preview button to manually refresh camera preview
- Auto-populate Add Camera dialog from discovered cameras
- /api/test-capture endpoint for URL preview

### Improved
- Better discovery item layout with preview area
- Parse discovered URL to populate all builder fields

## [1.2.6] - 2026

### Added
- Auto-load first camera when opening Live Preview
- Move ROI by dragging inside the rectangle
- Resize ROI by dragging corner handles
- Cursor changes to indicate move/resize/draw mode

### Improved
- OCR accuracy significantly improved with multiple preprocessing attempts
- CLAHE contrast enhancement for digital displays
- Denoising for cleaner images
- Try multiple PSM modes (single line, single word, block, raw)
- Scale up small ROIs more aggressively (3-4x for tiny regions)
- Better handling of dark displays

## [1.2.5] - 2026

### Added
- Value history display on dashboard (last 20 readings per camera)
- History tabs to switch between cameras
- Unit preset dropdown with common units (°C, °F, bar, psi, %, V, W, etc.)
- Port scanning for camera discovery (finds cameras without ONVIF support)
- Improved camera discovery with both ONVIF and port scanning

### Changed
- Unit input changed from text field to dropdown with presets

## [1.2.4] - 2026

### Added
- Zoom controls in Live Preview (mouse wheel or +/- buttons, up to 400%)
- Rotation controls in Live Preview (rotate left/right buttons)
- Zoom level indicator

### Fixed
- Fixed ROI rectangle drawing offset - now starts at exact cursor position
- Improved mouse position calculation for ROI selection

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
