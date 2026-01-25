#!/usr/bin/env python3
"""Camera Data Extractor Add-on Server."""

import os
import json
import logging
import base64
import time
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np
import pytesseract
from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS

# Configure logging
log_level = os.environ.get('LOG_LEVEL', 'info').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Paths
OPTIONS_PATH = os.environ.get('OPTIONS_PATH', '/data/options.json')
DATA_PATH = os.environ.get('DATA_PATH', '/data')
TEMPLATES_PATH = os.environ.get('TEMPLATES_PATH', '/data/templates')

# Ensure directories exist
Path(TEMPLATES_PATH).mkdir(parents=True, exist_ok=True)


@dataclass
class CameraConfig:
    """Camera configuration."""
    name: str
    stream_url: str
    username: str = ""
    password: str = ""
    value_name: str = "Value"
    unit: str = ""
    roi_x: int = 0
    roi_y: int = 0
    roi_width: int = 0
    roi_height: int = 0
    preprocessing: str = "auto"
    template_name: str = ""


@dataclass
class ExtractedValue:
    """Extracted value from camera."""
    camera_name: str
    value: float | None
    raw_text: str
    confidence: float
    timestamp: float
    error: str | None = None


class CameraProcessor:
    """Process camera streams and extract values."""

    def __init__(self):
        self.cameras: dict[str, CameraConfig] = {}
        self.values: dict[str, ExtractedValue] = {}
        self.templates: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def load_config(self):
        """Load configuration from options.json."""
        try:
            with open(OPTIONS_PATH, 'r') as f:
                options = json.load(f)

            self.cameras.clear()
            for cam_config in options.get('cameras', []):
                camera = CameraConfig(
                    name=cam_config.get('name', 'Camera'),
                    stream_url=cam_config.get('stream_url', ''),
                    username=cam_config.get('username', ''),
                    password=cam_config.get('password', ''),
                    value_name=cam_config.get('value_name', 'Value'),
                    unit=cam_config.get('unit', ''),
                    roi_x=cam_config.get('roi_x', 0),
                    roi_y=cam_config.get('roi_y', 0),
                    roi_width=cam_config.get('roi_width', 0),
                    roi_height=cam_config.get('roi_height', 0),
                    preprocessing=cam_config.get('preprocessing', 'auto'),
                    template_name=cam_config.get('template_name', ''),
                )
                self.cameras[camera.name] = camera
                logger.info(f"Loaded camera: {camera.name}")

            return options.get('scan_interval', 30)

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return 30

    def get_authenticated_url(self, camera: CameraConfig) -> str:
        """Build authenticated stream URL."""
        if not camera.username or not camera.password:
            return camera.stream_url

        if "@" in camera.stream_url:
            return camera.stream_url

        if "://" in camera.stream_url:
            protocol, rest = camera.stream_url.split("://", 1)
            return f"{protocol}://{camera.username}:{camera.password}@{rest}"

        return camera.stream_url

    def capture_frame(self, camera: CameraConfig) -> tuple[np.ndarray | None, str | None]:
        """Capture a frame from camera."""
        try:
            url = self.get_authenticated_url(camera)
            cap = cv2.VideoCapture(url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                return None, "Failed to open stream"

            # Read a few frames to get latest
            for _ in range(3):
                ret, frame = cap.read()

            cap.release()

            if not ret or frame is None:
                return None, "Failed to read frame"

            return frame, None

        except Exception as e:
            return None, str(e)

    def preprocess_image(self, image: np.ndarray, method: str) -> np.ndarray:
        """Preprocess image for OCR."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Scale up small images
        h, w = gray.shape[:2]
        if h < 100 or w < 100:
            scale = max(100 / h, 100 / w, 2)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        if method == "none":
            return gray
        elif method == "threshold":
            _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return processed
        elif method == "adaptive":
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        elif method == "invert":
            _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return cv2.bitwise_not(processed)
        else:  # auto
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(gray) < 127 and np.mean(binary) > 127:
                return binary
            elif np.mean(gray) < 127:
                return cv2.bitwise_not(binary)
            return binary

    def extract_value(self, camera: CameraConfig, frame: np.ndarray) -> ExtractedValue:
        """Extract numeric value from frame."""
        try:
            # Apply ROI if specified
            if camera.roi_width > 0 and camera.roi_height > 0:
                x, y, w, h = camera.roi_x, camera.roi_y, camera.roi_width, camera.roi_height
                img_h, img_w = frame.shape[:2]
                x = max(0, min(x, img_w - 1))
                y = max(0, min(y, img_h - 1))
                w = min(w, img_w - x)
                h = min(h, img_h - y)
                frame = frame[y:y+h, x:x+w]

            # Preprocess
            processed = self.preprocess_image(frame, camera.preprocessing)

            # OCR
            config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.-"
            raw_text = pytesseract.image_to_string(processed, config=config).strip()

            # Get confidence
            data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
            confidences = [int(c) for c in data["conf"] if str(c).isdigit() and int(c) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            # Parse value
            import re
            value = None
            for pattern in [r"-?\d+\.\d+", r"-?\d+,\d+", r"-?\d+"]:
                match = re.search(pattern, raw_text)
                if match:
                    try:
                        value = float(match.group().replace(",", "."))
                        break
                    except ValueError:
                        continue

            return ExtractedValue(
                camera_name=camera.name,
                value=round(value, 1) if value is not None else None,
                raw_text=raw_text,
                confidence=avg_confidence,
                timestamp=time.time(),
            )

        except Exception as e:
            return ExtractedValue(
                camera_name=camera.name,
                value=None,
                raw_text="",
                confidence=0,
                timestamp=time.time(),
                error=str(e),
            )

    def process_camera(self, camera: CameraConfig):
        """Process a single camera."""
        frame, error = self.capture_frame(camera)

        if error:
            self.values[camera.name] = ExtractedValue(
                camera_name=camera.name,
                value=None,
                raw_text="",
                confidence=0,
                timestamp=time.time(),
                error=error,
            )
            return

        result = self.extract_value(camera, frame)
        self.values[camera.name] = result

        if result.value is not None:
            logger.info(f"{camera.name}: {result.value} {camera.unit} (confidence: {result.confidence:.1f}%)")
        else:
            logger.warning(f"{camera.name}: Failed to extract value - {result.error or result.raw_text}")

    def run_loop(self, scan_interval: int):
        """Run the processing loop."""
        self._running = True
        while self._running:
            for camera in self.cameras.values():
                if not self._running:
                    break
                try:
                    self.process_camera(camera)
                except Exception as e:
                    logger.error(f"Error processing {camera.name}: {e}")

            # Wait for next scan
            for _ in range(scan_interval * 10):
                if not self._running:
                    break
                time.sleep(0.1)

    def start(self):
        """Start the processor."""
        scan_interval = self.load_config()
        self._thread = threading.Thread(target=self.run_loop, args=(scan_interval,), daemon=True)
        self._thread.start()
        logger.info(f"Started camera processor with {len(self.cameras)} cameras, interval: {scan_interval}s")

    def stop(self):
        """Stop the processor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)


# Global processor instance
processor = CameraProcessor()


# Web UI Template
WEB_UI = '''
<!DOCTYPE html>
<html>
<head>
    <title>Camera Data Extractor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1c1c1c; color: #e0e0e0; padding: 20px; }
        h1 { margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
        .card { background: #2d2d2d; border-radius: 8px; padding: 20px; margin-bottom: 16px; }
        .card-title { font-size: 18px; font-weight: 500; margin-bottom: 12px; color: #03a9f4; }
        .value { font-size: 48px; font-weight: bold; color: #4caf50; }
        .unit { font-size: 24px; color: #a0a0a0; }
        .meta { font-size: 12px; color: #757575; margin-top: 8px; }
        .error { color: #f44336; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
        .btn { background: #03a9f4; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #0288d1; }
        .preview { max-width: 100%; border-radius: 4px; margin-top: 12px; }
        .refresh { position: fixed; bottom: 20px; right: 20px; }
    </style>
</head>
<body>
    <h1>📷 Camera Data Extractor</h1>
    <div id="cameras" class="grid"></div>
    <button class="btn refresh" onclick="refresh()">↻ Refresh</button>
    <script>
        async function loadData() {
            const resp = await fetch('/api/values');
            const data = await resp.json();
            const container = document.getElementById('cameras');
            container.innerHTML = '';

            if (Object.keys(data).length === 0) {
                container.innerHTML = '<div class="card">No cameras configured. Add cameras in the add-on configuration.</div>';
                return;
            }

            for (const [name, info] of Object.entries(data)) {
                const card = document.createElement('div');
                card.className = 'card';
                card.innerHTML = `
                    <div class="card-title">${name}</div>
                    <div class="value ${info.error ? 'error' : ''}">${info.value !== null ? info.value : '--'}</div>
                    <div class="unit">${info.unit || ''}</div>
                    <div class="meta">
                        ${info.error ? '<span class="error">Error: ' + info.error + '</span><br>' : ''}
                        Raw: "${info.raw_text || ''}" | Confidence: ${(info.confidence || 0).toFixed(1)}%<br>
                        Updated: ${new Date(info.timestamp * 1000).toLocaleTimeString()}
                    </div>
                `;
                container.appendChild(card);
            }
        }

        function refresh() { loadData(); }
        loadData();
        setInterval(loadData, 5000);
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """Serve web UI."""
    return render_template_string(WEB_UI)


@app.route('/api/values')
def get_values():
    """Get all extracted values."""
    result = {}
    for name, value in processor.values.items():
        camera = processor.cameras.get(name)
        result[name] = {
            **asdict(value),
            'unit': camera.unit if camera else '',
        }
    return jsonify(result)


@app.route('/api/cameras')
def get_cameras():
    """Get camera configurations."""
    return jsonify({name: asdict(cam) for name, cam in processor.cameras.items()})


@app.route('/api/capture/<camera_name>')
def capture_frame(camera_name):
    """Capture a frame from a camera."""
    camera = processor.cameras.get(camera_name)
    if not camera:
        return jsonify({'error': 'Camera not found'}), 404

    frame, error = processor.capture_frame(camera)
    if error:
        return jsonify({'error': error}), 500

    _, buffer = cv2.imencode('.png', frame)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')

    h, w = frame.shape[:2]
    return jsonify({
        'success': True,
        'frame': frame_b64,
        'width': w,
        'height': h,
    })


@app.route('/api/reload', methods=['POST'])
def reload_config():
    """Reload configuration."""
    processor.load_config()
    return jsonify({'success': True, 'cameras': len(processor.cameras)})


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'cameras': len(processor.cameras)})


if __name__ == '__main__':
    logger.info("Starting Camera Data Extractor Add-on")
    processor.start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
