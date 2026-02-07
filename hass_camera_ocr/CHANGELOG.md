# Changelog

## [1.2.46] - 2026

### Improved
- **OCR accuracy for 7-segment digital displays** (combi boilers, gauges, meters):
  - **Preprocessing**: INTER_LANCZOS4 upscaling for sharper digit edges, larger morphological kernels (3x3) to bridge 7-segment gaps, dilation to thicken thin segments, higher CLAHE contrast (clipLimit=4.0), smarter auto-inversion using border vs center brightness analysis
  - **Tesseract**: Added `--dpi 300`, tries both normal and inverted images for each preprocessing method, raised early-exit threshold from 80% to 90%
  - **EasyOCR**: Now tries multiple preprocessing methods (auto/threshold/invert) with normal+inverted variants, added `text_threshold=0.3` and `low_text=0.3` for better small digit detection
  - **PaddleOCR**: Now tries multiple preprocessing methods with normal+inverted variants

## [1.2.45] - 2026

### Fixed
- **Test Extract button in Saved ROI Details dialog**: Fixed broken onclick handler caused by unescaped JSON in HTML attributes
- **Apply ROI button**: Same HTML escaping fix applied

### Added
- **Per-provider OCR results in ROI Details dialog**: Shows results from all configured OCR providers after Test Extract, with selected provider highlighted
- **In-place test results**: Test Extract updates the dialog in-place instead of closing it

## [1.2.44] - 2026

### Changed
- **OCR providers configured for numbers only**:
  - Tesseract: Uses character whitelist `0123456789.-`
  - EasyOCR: Uses `allowlist='0123456789.-'` parameter
  - PaddleOCR: Filters output to keep only numeric characters
  - Cloud providers: Results filtered for numbers only
  - AI providers: Prompt updated to "extract numeric value only"
- All providers now consistently return positive values

## [1.2.43] - 2026

### Changed
- **OCR values are always positive**: Minus signs in OCR results are now removed - all extracted values are 0 or greater
  - Applies to Tesseract, TrOCR (ML), and all other OCR providers
  - Prevents misread display segments from being interpreted as negative numbers

## [1.2.42] - 2026

### Fixed
- **Scroll on buttons no longer changes zoom**: Mouse wheel over buttons, inputs, and controls now behaves normally instead of zooming the preview

## [1.2.41] - 2026

### Fixed
- **Test Extract Button Not Working**: Fixed JavaScript error where `currentImage` variable was undefined - now correctly uses `previewImage` and the preview img element

## [1.2.40] - 2026

### Added
- **Test All ROIs - Provider Results Display**: Each ROI card now shows results from all OCR providers after Test All completes
  - Expandable provider results section below each ROI card
  - Shows provider name, extracted value, and confidence percentage
  - Best provider highlighted with ★ indicator
  - Click "▼ X providers" to expand/collapse results
  - Results auto-expand after testing

## [1.2.39] - 2026

### Added
- **Test Extract Improvements**:
  - Captured image now shows immediately before OCR processing starts
  - All provider results displayed below the best result with confidence percentages
  - Shows which provider was selected as "BEST" with a star indicator

- **Interactive ROI Rotation**: Drag the rotation handle (circle above ROI) to rotate the rectangle on the live preview canvas
  - Visual rotation handle with curved arrow icon
  - Snaps to 15° increments when dragging near those angles
  - Rotation angle displayed above the ROI

- **Test All ROIs Progress Indicator**:
  - Currently processing ROI is highlighted with pulsing blue border
  - Auto-scrolls to show the active ROI
  - Progress counter shows "Testing X/Y..."
  - Each ROI border color indicates result quality (green/yellow/red)
  - Loading spinner shown while extracting each ROI

## [1.2.38] - 2026

### Fixed
- **OCR Settings Save Bug**: Fixed issue where OCR provider configurations were not being saved correctly
  - Added deep copy of provider arrays/objects in UI to prevent reference issues
  - Added proper API key masking in server responses with `has_api_key` flag
  - Server now preserves existing API keys when masked values are sent back
  - Provider configuration form now shows "configured" indicator when API key exists
  - Changed toast message to clarify that provider settings need "Save Configuration" to persist

### Improved
- Added debug logging throughout save/load flow to help diagnose configuration issues
- API key security: keys in `provider_configs` are now masked in API responses
- Better UX: shows "Leave empty to keep existing key" hint when editing configured provider

## [1.2.37] - 2026

### Fixed
- Fixed OCR providers not loading from saved camera config - `ocr_providers`, `provider_configs`, and all AI-related fields now properly loaded when camera is loaded from persistent storage
- Test All ROIs now correctly uses camera's configured OCR providers

## [1.2.36] - 2026

### Added
- **Rotated ROI Rectangle Visualization**: The blue ROI rectangle now visually rotates on the canvas when rotation is set
  - Both EDIT and TEST modes show the rotated rectangle
  - Corner handles rotate with the rectangle
  - Dark overlay mask properly clips the rotated ROI area
  - Center point indicator shows rotation pivot
  - Arc and angle label show rotation direction and amount

## [1.2.35] - 2026

### Changed
- **OCR Config moved to Live Preview**: Removed separate AI Settings tab, added "OCR Config" button in Live Preview
  - Click OCR Config button to open configuration popup for the selected camera
  - Configure providers, drag to reorder, set API keys all from Live Preview
- **Test Extraction now uses camera's OCR providers**: Test Extract button uses all configured providers and shows which one returned the best result
- Extraction result now shows provider name alongside confidence percentage

### Fixed
- `extract_from_frame` now properly uses camera's configured OCR providers instead of only Tesseract
- ROI rotation is now applied during test extraction

## [1.2.34] - 2026

### Added
- **ROI Rotation Support**: Rotate the Region of Interest before OCR extraction
  - Rotate buttons (90° left/right) in Live Preview
  - Custom rotation angle input (-180° to 180°)
  - Visual rotation indicator on ROI overlay
  - Rotation stored in camera config and history
- **Improved PTZ for RTSP Cameras**: PTZ now tries multiple ONVIF ports (80, 8080, 8000, 8899, 2020, 8081, 8088) to find the correct endpoint

### Fixed
- PTZ pan/tilt buttons now work with RTSP stream URLs by probing common ONVIF HTTP ports

## [1.2.33] - 2026

### Fixed
- Fixed NameError crash on startup: moved logging configuration before optional OCR imports
- EasyOCR/PaddleOCR import failures now log correctly instead of crashing

## [1.2.32] - 2026

### Fixed
- Fixed Docker build failure on aarch64/armv7/armhf: EasyOCR and PaddleOCR now only install on amd64/i386 where pre-built wheels are available
- ARM devices (Raspberry Pi, etc.) will use Tesseract and ML (TrOCR) providers instead

## [1.2.31] - 2026

### Added
- **EasyOCR Provider**: New local OCR engine using deep learning, supports multiple languages and fonts
- **PaddleOCR Provider**: High-accuracy OCR engine by Baidu, excellent for structured text and digital displays
- **OCR Providers Status API**: New `/api/ocr/providers` endpoint to check availability of all local OCR engines
- Tesseract, EasyOCR, PaddleOCR, and ML (TrOCR) now all shown explicitly in provider selection UI

### Technical
- EasyOCR uses lazy loading to minimize startup time
- PaddleOCR configured with CPU-only PaddlePaddle for compatibility
- Both new engines support the existing preprocessing pipeline

## [1.2.30] - 2026

### Fixed
- Fixed "Failed to open stream" error for HTTP image URLs (snapshot URLs like `/snapshot.jpg`)
- Added missing `requests` import that caused discovery and HTTP capture to fail
- HTTP image URLs now use `requests.get()` for better authentication and static image support
- HTTP video streams (MJPEG) now use FFmpeg backend for improved compatibility
- Better error messages for HTTP connection issues (timeout, auth failure, 404)

## [1.2.29] - 2026

### Fixed
- Fixed "Camera name or image required" error when testing AI providers with camera source
- Test All Providers now correctly passes camera name to all provider tests when using camera source

## [1.2.28] - 2026

### Fixed
- Fixed Dockerfile build error on aarch64: use Alpine's native `py3-scikit-learn` package instead of pip (avoids compilation requiring gcc)
- Reorganized Dockerfile with separate RUN commands for better caching

## [1.2.27] - 2026

### Fixed
- Fixed Dockerfile build error: separated torch installation with correct `--index-url` for CPU-only version

## [1.2.26] - 2026

### Added
- **Multi-Provider Configuration UI**: New settings page to configure multiple OCR providers per camera
  - Drag-and-drop provider ordering
  - Per-provider credential configuration (API keys, endpoints)
  - Visual provider status indicators
  - Test button to run all configured providers
- **Provider Results in History Dialog**: Click history entry to see results from all providers with confidence scores

### Changed
- Replaced AI Settings page with unified Multi-Provider Configuration
- Provider execution order now user-configurable via drag-and-drop

## [1.2.25] - 2026

### Added
- **ML-based OCR Provider**: New provider using HuggingFace TrOCR model for text extraction
- **ML ROI Locator**: Uses CLIP embeddings to automatically locate ROI in frames based on trained examples
- **Multi-Provider Support**: Configure multiple OCR providers per camera in preferred order
  - All providers run in sequence, results collected with confidence scores
  - Best result selected automatically based on confidence
  - All provider results stored in history for comparison
- **Provider Results in History**: Value history now shows results from all providers, not just the selected one
- New API endpoints: `/api/ml/status`, `/api/ml/test`, `/api/ml/locate-roi`
- ML models persisted in `/config/hass_camera_ocr/ml_models/` (survives reinstalls)

### Changed
- Train OCR now also trains ML models (ROI Locator and Text Extractor)
- Camera configuration now supports `ocr_providers` list and `provider_configs` object
- Updated README with new architecture diagram and multi-provider documentation

### Technical
- Added torch, transformers, Pillow, scikit-learn dependencies
- Added MLService class with CLIP and TrOCR model management
- Added ProviderResult dataclass for structured provider results
- Updated ExtractedValue to include provider_results and selected_provider

## [1.2.24] - 2026

### Added
- **Per-Camera AI Configuration**: Each camera can now have its own AI provider settings
- AI Settings page now shows camera selector to configure AI per camera
- Support for different AI/OCR providers per camera
- **Clear History Button**: Clear value history per camera from the dashboard

### Changed
- Moved HTML from server.py to separate template file (templates/index.html)
- Unified saved ROIs with camera ROI config - "Save ROI" now saves to both
- Live preview area now maintains 16:9 aspect ratio
- Removed separate "Save ROI to Camera" button (Apply does this now)

### Fixed
- Improved code maintainability with separate template files
- Fixed "No saved ROIs for this camera" error in Train OCR and Validate ROI functions
- Fixed AI test using form values instead of requiring saved config first
- Fixed Train OCR showing 0% accuracy due to strict string comparison (now uses flexible numeric comparison)
- Fixed test extraction using incorrect coordinates when zoomed/scrolled
- Fixed "Test All" button clearing validation values in ROI boxes (now only updates extracted value)

## [1.2.19] - 2026

### Changed
- Version bump to trigger add-on update

## [1.2.18] - 2026

### Added
- **OCR Training Feature**: Validate saved ROIs with correct values to improve OCR accuracy
  - Click "Validate" on any saved ROI to enter the correct numeric value
  - Use "Train OCR" button to test different preprocessing settings on validated ROIs
  - Shows optimal preprocessing and PSM mode configuration based on training results
- **Validated ROI Badges**: Green checkmark on saved ROIs that have been validated
- **Saved ROI Detail Dialog**: Click on any saved ROI to view full details with Apply/Test/Validate buttons
- **ROI Test Busy Indicator**: Visual feedback when testing saved ROIs

### Fixed
- Dashboard now shows cameras waiting for first extraction instead of "No cameras configured"
- Dashboard status shows "Waiting" for cameras that haven't been scanned yet

## [1.2.17] - 2026

### Changed
- Version bump to trigger add-on update

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
