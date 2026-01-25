# CLAUDE.md - Project Guidelines for Camera OCR

## Project Overview

**Camera OCR** (formerly Camera Data Extractor) is a Home Assistant add-on and custom integration that extracts numeric values from IP camera video streams using OCR (Optical Character Recognition). It's designed for monitoring displays like boiler temperatures, pressure gauges, energy meters, or any device with a digital/analog display.

**Repository**: https://github.com/tursoft/hass-camera-ocr
**Author**: Muhammet Turşak (tursoft@gmail.com)

## Project Structure

```
hass-camera-ocr/
├── hass_camera_ocr/              # Home Assistant Add-on
│   ├── app/
│   │   └── server.py             # Main Flask application (Web UI + API + OCR processing)
│   ├── rootfs/
│   │   └── etc/
│   │       ├── nginx/nginx.conf  # Nginx reverse proxy config
│   │       └── services.d/       # s6-overlay service scripts
│   ├── config.yaml               # Add-on configuration schema
│   ├── build.yaml                # Multi-arch build configuration
│   ├── Dockerfile                # Container build instructions
│   ├── CHANGELOG.md              # Version history
│   └── DOCS.md                   # Add-on documentation
├── custom_components/
│   └── hass_camera_ocr/          # HACS Custom Integration
│       ├── __init__.py
│       ├── manifest.json
│       ├── sensor.py
│       └── ...
├── .github/workflows/
│   └── build.yaml                # CI/CD for multi-arch Docker builds
├── README.md                     # Main documentation
├── repository.json               # Add-on repository metadata
└── hacs.json                     # HACS integration metadata
```

## Architecture

### Add-on Architecture

1. **Flask Web Server** (`server.py`):
   - Serves the admin web UI (single-page application embedded in Python)
   - REST API endpoints for camera management, values, history, templates
   - OCR processing engine using OpenCV and Tesseract
   - Template matching for ROI tracking across camera movements
   - ONVIF discovery + port scanning for camera detection

2. **Nginx Reverse Proxy**:
   - Handles ingress traffic from Home Assistant
   - Proxies requests to Flask on port 5000

3. **s6-overlay Services**:
   - `camera-extractor`: Runs the Flask application
   - `nginx`: Runs the reverse proxy

4. **Persistent Storage**:
   - Camera config stored in `/config/hass_camera_ocr/cameras.json`
   - Templates stored in `/config/hass_camera_ocr/templates/`
   - Survives add-on uninstall/reinstall

### Key Components in server.py

| Component | Purpose |
|-----------|---------|
| `CameraConfig` | Dataclass for camera configuration |
| `ExtractedValue` | Dataclass for OCR results |
| `TemplateMatcher` | ROI tracking with scale/rotation invariance |
| `ONVIFDiscovery` | WS-Discovery for ONVIF cameras |
| `PortScanner` | Port scanning for non-ONVIF cameras |
| `CameraProcessor` | Main processing loop, capture, OCR extraction |

### Web UI Pages

1. **Dashboard**: Live extracted values, value history (last 20 readings)
2. **Cameras**: CRUD operations for camera management
3. **Live Preview**: ROI selection with zoom/rotation, visual feedback
4. **Templates**: Template management for auto-tracking ROI
5. **Discovery**: Network scan for cameras (ONVIF + port scan)

## Development Rules

### Version Management

- **Always update these files when version changes**:
  - `hass_camera_ocr/config.yaml` (version field)
  - `hass_camera_ocr/CHANGELOG.md` (add new version section)
  - `custom_components/hass_camera_ocr/manifest.json` (version field)
  - `README.md` (version badge)

### Auto Version Bump & Git Push

**IMPORTANT: After each successfully completed prompt/task, Claude MUST:**

1. Bump the version number (increment patch version)
2. Update all version files listed above
3. Add changelog entry describing the changes
4. Commit all changes with descriptive message
5. Push to git

**Manual script available**: `scripts/bump-version.ps1 -Message "description"`

**Or do it manually**:
```bash
# 1. Increment patch version in all files (e.g., 1.2.9 -> 1.2.10)
# 2. Add changelog entry
# 3. Commit and push
git add -A
git commit -m "Feature description (v1.2.10)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
git push
```

### Commit Conventions

- Auto commit and push after each completed task
- Use descriptive commit messages with version number
- Include `Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>`

### Code Style

- Single-file architecture for `server.py` (embedded HTML/CSS/JS)
- Use relative API paths (`api/...` not `/api/...`) for Home Assistant ingress compatibility
- All CSS in `<style>` block, all JS in `<script>` block within the HTML template

### Docker/Add-on

- Use `opencv-python-headless` via pip (not Alpine's py3-opencv)
- Include FFmpeg for RTSP stream support
- Map `/config:rw` and `/addon_config:rw` for persistent storage

## User Requirements Summary

### Core Functionality
- Extract numeric values from IP camera streams using OCR
- Support RTSP and HTTP video streams
- Define Region of Interest (ROI) for extraction
- Template matching for auto-tracking ROI when camera moves/rotates

### UI/UX Requirements
1. **Add Camera Dialog**:
   - Toggle between "Full URL" and "Build URL" modes
   - Bidirectional URL parsing (URL ↔ components)
   - Unit preset dropdown (°C, °F, bar, psi, %, V, W, etc.) - easy to select common units

2. **Live Preview**:
   - Zoom controls (mouse wheel, +/- buttons, up to 400%)
   - Rotation controls (rotate left/right by 90°)
   - ROI rectangle drawing at exact cursor position
   - Visual overlay showing ROI borders and extracted value

3. **Dashboard**:
   - Live extracted values display
   - Value history table (last 20 readings per camera)
   - Tabs to switch between cameras in history view

4. **Camera Discovery**:
   - ONVIF WS-Discovery for compatible cameras
   - Port scanning fallback for non-ONVIF cameras
   - Combined results from both methods

### Technical Requirements
- Persistent storage survives uninstall/reinstall
- FFmpeg support for RTSP streams
- TCP transport for reliable RTSP connections
- Connection timeouts to prevent hanging
- Detailed logging for debugging

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/cameras` | GET | List all cameras |
| `/api/cameras` | POST | Add new camera |
| `/api/cameras/<name>` | PUT | Update camera |
| `/api/cameras/<name>` | DELETE | Delete camera |
| `/api/values` | GET | Get current extracted values |
| `/api/history` | GET | Get value history for all cameras |
| `/api/history/<name>` | GET | Get history for specific camera |
| `/api/capture/<name>` | GET | Capture frame from camera |
| `/api/templates` | GET | List templates |
| `/api/templates` | POST | Save template |
| `/api/templates/<name>` | DELETE | Delete template |
| `/api/discover` | POST | Discover cameras on network |
| `/api/test-extraction` | POST | Test OCR on uploaded image |

## Common Issues & Solutions

### RTSP Stream Fails
- Add FFmpeg to Dockerfile
- Use TCP transport: `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`
- Add connection timeouts

### API Returns 404
- Use relative paths (`api/...`) not absolute (`/api/...`)
- Required for Home Assistant ingress compatibility

### Camera Discovery Finds Nothing
- ONVIF multicast may not work in Docker
- Port scanner added as fallback
- Check network connectivity between container and cameras

### ROI Drawing Offset
- Fixed with proper mouse position calculation
- Account for CSS transforms and zoom level

## Version History Highlights

| Version | Key Changes |
|---------|-------------|
| 1.0.0 | Initial release |
| 1.0.1 | OpenCV/s6 fixes for Alpine Linux |
| 1.1.0 | Full admin web UI, visual ROI selection, template matching |
| 1.1.1 | Fix API URLs for ingress compatibility |
| 1.1.2 | Persistent storage in /config directory |
| 1.2.0 | Renamed to Camera OCR, repository to hass-camera-ocr |
| 1.2.1 | URL Builder with bidirectional parsing |
| 1.2.2 | FFmpeg support for RTSP streams |
| 1.2.4 | Zoom/rotation controls, ROI drawing fix |
| 1.2.5 | Value history, unit presets, improved discovery |
| 1.2.6 | ROI move/resize, improved OCR accuracy |
| 1.2.7 | Discovery previews, auto-populate Add Camera dialog |
| 1.2.8 | PTZ controls, AI integration (OpenAI, Anthropic, Google, Ollama) |
| 1.2.9 | Version bump, auto-bump script added |

## Testing Checklist

Before releasing:
- [ ] Add-on installs without errors
- [ ] Web UI loads via ingress
- [ ] Camera can be added and configured
- [ ] Live preview captures frames
- [ ] ROI selection works correctly
- [ ] OCR extracts values
- [ ] Values display on dashboard
- [ ] History updates with new readings
- [ ] Camera discovery finds cameras
- [ ] Settings persist after add-on restart
- [ ] Settings persist after add-on reinstall
