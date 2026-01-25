# Camera Data Extractor for Home Assistant

Extract numeric values from IP camera video streams using OCR (Optical Character Recognition). Perfect for monitoring displays like boiler temperatures, pressure gauges, energy meters, or any device with a digital/analog display.

## Features

- **Video Stream Support**: Connect to RTSP and HTTP video streams from IP cameras
- **OCR Value Extraction**: Automatically extract numeric values from camera images
- **Region of Interest (ROI)**: Define specific areas of the frame to analyze
- **Template Matching**: Handle camera movement/rotation by using reference templates
- **Auto-Discovery**: Discover ONVIF-compatible cameras on your network
- **Web UI Panel**: Easy-to-use interface for configuration and ROI selection
- **Home Assistant Integration**: Exposes values as sensor entities for automations and monitoring

## Installation

### HACS (Recommended)

1. Open HACS in Home Assistant
2. Click the three dots menu (⋮) → Custom repositories
3. Add `https://github.com/your-username/ha-camera-data-extractor` as an Integration
4. Search for "Camera Data Extractor" and install
5. Restart Home Assistant

### Manual Installation

1. Download the latest release from GitHub
2. Extract and copy the `custom_components/camera_data_extractor` folder to your Home Assistant's `custom_components` directory
3. Restart Home Assistant

## Prerequisites

This integration requires **Tesseract OCR** to be installed on your Home Assistant system.

### Home Assistant OS / Supervised

Tesseract should be available automatically. If not, you may need to install it via the terminal add-on:

```bash
apk add tesseract-ocr
```

### Home Assistant Container

Add Tesseract to your Docker image or mount it from the host:

```bash
apt-get install tesseract-ocr
```

### Home Assistant Core

Install Tesseract on your system:

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Fedora
sudo dnf install tesseract

# macOS
brew install tesseract
```

## Configuration

### Adding a Camera

1. Go to **Settings** → **Devices & Services** → **Add Integration**
2. Search for "Camera Data Extractor"
3. Enter your camera details:
   - **Camera Name**: A friendly name for your camera
   - **Stream URL**: The RTSP or HTTP URL of your camera stream
     - RTSP example: `rtsp://192.168.1.100:554/stream1`
     - HTTP example: `http://192.168.1.100/video`
   - **Username/Password**: Optional credentials for camera authentication

4. Configure extraction settings:
   - **Value Name**: Name for the extracted value (e.g., "Temperature")
   - **Unit of Measurement**: Unit for display (e.g., "°C", "bar", "%")
   - **Scan Interval**: How often to capture and analyze frames (seconds)
   - **Preprocessing**: Image preprocessing method for better OCR accuracy

5. Set up the Region of Interest (ROI):
   - Set coordinates to define the area containing the value
   - Use the Web UI panel for visual selection (recommended)

### Using the Web UI Panel

After installation, a new **Camera Data Extractor** panel appears in your Home Assistant sidebar.

1. **Cameras Tab**: View configured cameras and capture live frames
2. **ROI Selection Tab**:
   - Click and drag on the frame to select the region containing the value
   - Fine-tune using the coordinate inputs
   - Save as a template for cameras that may move
3. **Templates Tab**: Manage saved templates and test template matching
4. **Discovery Tab**: Scan your network for ONVIF cameras

### Template Matching (For Moving Cameras)

If your camera can pan/tilt/zoom, use template matching to find the correct region even when the camera moves:

1. Position your camera to show the display
2. Go to the ROI Selection tab
3. Select the region containing the value
4. Enter a template name and click "Save as Template"
5. Use the `camera_data_extractor.use_template` service to update the ROI based on the current camera position

## Common Camera Stream URLs

| Brand | Typical RTSP URL |
|-------|-----------------|
| Hikvision | `rtsp://user:pass@IP:554/Streaming/Channels/101` |
| Dahua | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0` |
| Reolink | `rtsp://user:pass@IP:554/h264Preview_01_main` |
| Amcrest | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0` |
| Ubiquiti | `rtsp://user:pass@IP:554/live` |
| Generic ONVIF | `rtsp://user:pass@IP:554/stream1` |

## Services

### `camera_data_extractor.capture_frame`

Immediately capture a frame and update the sensor value.

```yaml
service: camera_data_extractor.capture_frame
data:
  entity_id: sensor.boiler_temperature
```

### `camera_data_extractor.update_roi`

Update the Region of Interest coordinates.

```yaml
service: camera_data_extractor.update_roi
data:
  entity_id: sensor.boiler_temperature
  roi_x: 100
  roi_y: 50
  roi_width: 200
  roi_height: 80
```

### `camera_data_extractor.use_template`

Find and apply ROI from a saved template (for moving cameras).

```yaml
service: camera_data_extractor.use_template
data:
  entity_id: sensor.boiler_temperature
  template_name: boiler_display
```

## Sensor Attributes

Each sensor provides these attributes:

| Attribute | Description |
|-----------|-------------|
| `raw_text` | Raw text extracted by OCR |
| `confidence` | OCR confidence percentage |
| `last_frame_time` | Timestamp of last analyzed frame |
| `camera_name` | Name of the camera |
| `roi_x`, `roi_y`, `roi_width`, `roi_height` | Current ROI coordinates |
| `preprocessing` | Preprocessing method used |
| `stream_url` | Camera stream URL |

## Automation Examples

### Alert on High Temperature

```yaml
automation:
  - alias: "Boiler Temperature Alert"
    trigger:
      - platform: numeric_state
        entity_id: sensor.boiler_temperature
        above: 80
    action:
      - service: notify.mobile_app
        data:
          title: "Boiler Alert"
          message: "Temperature is {{ states('sensor.boiler_temperature') }}°C!"
```

### Log Temperature Changes

```yaml
automation:
  - alias: "Log Boiler Temperature"
    trigger:
      - platform: state
        entity_id: sensor.boiler_temperature
    action:
      - service: logbook.log
        data:
          name: Boiler
          message: "Temperature changed to {{ states('sensor.boiler_temperature') }}°C"
```

### Update ROI for PTZ Camera

```yaml
automation:
  - alias: "Update Boiler Camera ROI"
    trigger:
      - platform: time_pattern
        minutes: "/5"
    action:
      - service: camera_data_extractor.use_template
        data:
          entity_id: sensor.boiler_temperature
          template_name: boiler_display
```

## Preprocessing Options

| Option | Description | Best For |
|--------|-------------|----------|
| `auto` | Automatically determines best method | General use |
| `none` | No preprocessing | Clear, high-contrast displays |
| `threshold` | Binary threshold (Otsu's method) | LCD/LED displays |
| `adaptive` | Adaptive threshold | Varying lighting conditions |
| `invert` | Inverted colors | Light text on dark background |

## Troubleshooting

### OCR Not Reading Values Correctly

1. **Adjust ROI**: Make sure the ROI tightly frames only the numeric display
2. **Try Different Preprocessing**: Switch between `auto`, `threshold`, and `adaptive`
3. **Improve Lighting**: Ensure adequate lighting on the display
4. **Increase Resolution**: Use a higher resolution stream if available
5. **Check Decimal Places**: Adjust the decimal places setting

### Camera Connection Issues

1. **Verify Stream URL**: Test the URL in VLC or another media player
2. **Check Credentials**: Ensure username/password are correct
3. **Firewall Rules**: Make sure Home Assistant can reach the camera
4. **RTSP Port**: Default is 554, but some cameras use different ports

### Template Matching Not Working

1. **Retake Template**: Capture a new template in good lighting
2. **Check Confidence**: If confidence is low, the camera view may have changed too much
3. **Reduce Movement**: Template matching works best with small position changes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- [Report an Issue](https://github.com/your-username/ha-camera-data-extractor/issues)
- [Home Assistant Community Forum](https://community.home-assistant.io/)
