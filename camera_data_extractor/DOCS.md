# Camera Data Extractor Add-on

Extract numeric values from IP camera video streams using OCR (Optical Character Recognition).

## Features

- Connect to RTSP and HTTP video streams
- Extract numeric values using Tesseract OCR
- Define Region of Interest (ROI) for focused extraction
- Multiple image preprocessing options
- Web UI for monitoring values
- Home Assistant Ingress integration

## Configuration

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `cameras` | List of camera configurations | `[]` |
| `scan_interval` | Seconds between scans | `30` |
| `log_level` | Logging level | `info` |

### Camera Configuration

Each camera in the `cameras` list can have:

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique name for the camera |
| `stream_url` | Yes | RTSP or HTTP stream URL |
| `username` | No | Camera username |
| `password` | No | Camera password |
| `value_name` | No | Name for the extracted value |
| `unit` | No | Unit of measurement |
| `roi_x` | No | ROI X position (pixels) |
| `roi_y` | No | ROI Y position (pixels) |
| `roi_width` | No | ROI width (pixels) |
| `roi_height` | No | ROI height (pixels) |
| `preprocessing` | No | Image preprocessing method |

### Preprocessing Options

- `auto` - Automatically determine best method (default)
- `none` - No preprocessing
- `threshold` - Binary threshold
- `adaptive` - Adaptive threshold
- `invert` - Inverted colors (for light text on dark background)

## Example Configuration

```yaml
cameras:
  - name: Boiler Temperature
    stream_url: rtsp://192.168.1.100:554/stream1
    username: admin
    password: password123
    value_name: Temperature
    unit: °C
    roi_x: 100
    roi_y: 50
    roi_width: 200
    roi_height: 80
    preprocessing: auto
  - name: Pressure Gauge
    stream_url: http://192.168.1.101/video
    value_name: Pressure
    unit: bar
scan_interval: 30
log_level: info
```

## Finding Your Camera's Stream URL

### Common RTSP URLs by Brand

| Brand | URL Pattern |
|-------|-------------|
| Hikvision | `rtsp://user:pass@IP:554/Streaming/Channels/101` |
| Dahua | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0` |
| Reolink | `rtsp://user:pass@IP:554/h264Preview_01_main` |
| Generic ONVIF | `rtsp://user:pass@IP:554/stream1` |

### Tips for Better OCR

1. **Define a tight ROI** - Only include the numeric display area
2. **Ensure good lighting** - The display should be clearly visible
3. **Try different preprocessing** - If `auto` doesn't work, try `threshold` or `invert`
4. **Use higher resolution** - If available, use main stream instead of sub-stream

## Web Interface

Access the web interface through the add-on's "Open Web UI" button or via Home Assistant Ingress.

The interface shows:
- Current extracted value for each camera
- Raw OCR text
- Confidence percentage
- Last update timestamp

## API Endpoints

The add-on exposes REST API endpoints:

- `GET /api/values` - Get all extracted values
- `GET /api/cameras` - Get camera configurations
- `GET /api/capture/<camera_name>` - Capture a frame (base64 PNG)
- `POST /api/reload` - Reload configuration
- `GET /api/health` - Health check

## Troubleshooting

### OCR Not Reading Values

1. Check if the ROI correctly frames the numeric display
2. Try different preprocessing modes
3. Ensure adequate lighting
4. Check the confidence percentage - low values indicate recognition issues

### Connection Failed

1. Verify the stream URL in VLC or another media player
2. Check username/password
3. Ensure the camera is on the same network
4. Try both RTSP and HTTP URLs

### High CPU Usage

1. Increase `scan_interval` to reduce frequency
2. Use a smaller ROI
3. Use sub-stream instead of main stream (lower resolution)
