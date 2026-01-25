#!/usr/bin/env python3
"""Camera Data Extractor Add-on Server with Full Admin Interface and Template Matching."""

import os
import json
import logging
import base64
import time
import threading
import socket
import struct
import re
import uuid
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Any, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import pytesseract
from flask import Flask, jsonify, request, render_template_string, Response
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
Path(DATA_PATH).mkdir(parents=True, exist_ok=True)


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
    use_template_matching: bool = False


@dataclass
class ExtractedValue:
    """Extracted value from camera."""
    camera_name: str
    value: Optional[float]
    raw_text: str
    confidence: float
    timestamp: float
    roi_x: int = 0
    roi_y: int = 0
    roi_width: int = 0
    roi_height: int = 0
    error: Optional[str] = None


@dataclass
class DiscoveredCamera:
    """Discovered camera from network scan."""
    ip: str
    port: int
    manufacturer: str = ""
    model: str = ""
    name: str = ""
    stream_url: str = ""


class TemplateMatcher:
    """Template matching for ROI tracking across rotations and movements."""

    def __init__(self, templates_path: str):
        self.templates_path = Path(templates_path)
        self.templates: dict[str, dict] = {}
        self.load_templates()

    def load_templates(self):
        """Load all saved templates."""
        self.templates = {}
        for template_file in self.templates_path.glob("*.json"):
            try:
                with open(template_file, 'r') as f:
                    data = json.load(f)
                name = template_file.stem
                # Load the template image
                img_path = self.templates_path / f"{name}.png"
                if img_path.exists():
                    data['image'] = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    self.templates[name] = data
                    logger.info(f"Loaded template: {name}")
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")

    def save_template(self, name: str, frame: np.ndarray, roi: dict) -> bool:
        """Save a template for matching."""
        try:
            x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']

            # Extract ROI from frame
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            template_img = gray[y:y+h, x:x+w].copy()

            # Save template image
            img_path = self.templates_path / f"{name}.png"
            cv2.imwrite(str(img_path), template_img)

            # Save template metadata
            data = {
                'name': name,
                'original_roi': roi,
                'frame_size': {'width': frame.shape[1], 'height': frame.shape[0]},
                'created': time.time()
            }

            json_path = self.templates_path / f"{name}.json"
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)

            # Add to memory
            data['image'] = template_img
            self.templates[name] = data

            logger.info(f"Saved template: {name}")
            return True
        except Exception as e:
            logger.error(f"Error saving template {name}: {e}")
            return False

    def delete_template(self, name: str) -> bool:
        """Delete a template."""
        try:
            img_path = self.templates_path / f"{name}.png"
            json_path = self.templates_path / f"{name}.json"

            if img_path.exists():
                img_path.unlink()
            if json_path.exists():
                json_path.unlink()

            if name in self.templates:
                del self.templates[name]

            logger.info(f"Deleted template: {name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting template {name}: {e}")
            return False

    def find_template(self, frame: np.ndarray, template_name: str,
                      scales: list = None, rotations: list = None) -> Optional[dict]:
        """Find template in frame with scale and rotation invariance."""
        if template_name not in self.templates:
            return None

        template_data = self.templates[template_name]
        template_img = template_data.get('image')
        if template_img is None:
            return None

        # Convert frame to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Default scales and rotations
        if scales is None:
            scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        if rotations is None:
            rotations = [0, -5, 5, -10, 10, -15, 15, -30, 30, -45, 45, -90, 90, 180]

        best_match = None
        best_score = 0.0
        threshold = 0.6  # Minimum match threshold

        for scale in scales:
            for angle in rotations:
                try:
                    # Scale template
                    th, tw = template_img.shape[:2]
                    new_w = int(tw * scale)
                    new_h = int(th * scale)
                    if new_w < 10 or new_h < 10:
                        continue

                    scaled_template = cv2.resize(template_img, (new_w, new_h))

                    # Rotate template
                    if angle != 0:
                        center = (new_w // 2, new_h // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

                        # Calculate new bounds
                        cos = np.abs(rotation_matrix[0, 0])
                        sin = np.abs(rotation_matrix[0, 1])
                        new_w_rot = int(new_h * sin + new_w * cos)
                        new_h_rot = int(new_h * cos + new_w * sin)

                        rotation_matrix[0, 2] += (new_w_rot - new_w) / 2
                        rotation_matrix[1, 2] += (new_h_rot - new_h) / 2

                        rotated_template = cv2.warpAffine(scaled_template, rotation_matrix,
                                                          (new_w_rot, new_h_rot))
                    else:
                        rotated_template = scaled_template
                        new_w_rot, new_h_rot = new_w, new_h

                    # Skip if template is larger than image
                    if rotated_template.shape[0] > gray.shape[0] or \
                       rotated_template.shape[1] > gray.shape[1]:
                        continue

                    # Template matching
                    result = cv2.matchTemplate(gray, rotated_template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)

                    if max_val > best_score and max_val >= threshold:
                        best_score = max_val
                        best_match = {
                            'x': max_loc[0],
                            'y': max_loc[1],
                            'width': rotated_template.shape[1],
                            'height': rotated_template.shape[0],
                            'confidence': float(max_val),
                            'scale': scale,
                            'rotation': angle
                        }

                except Exception as e:
                    logger.debug(f"Template match error at scale={scale}, angle={angle}: {e}")
                    continue

        return best_match

    def get_template_list(self) -> list:
        """Get list of all templates."""
        result = []
        for name, data in self.templates.items():
            result.append({
                'name': name,
                'original_roi': data.get('original_roi', {}),
                'created': data.get('created', 0)
            })
        return result


class ONVIFDiscovery:
    """ONVIF WS-Discovery for cameras."""

    MULTICAST_IP = "239.255.255.250"
    MULTICAST_PORT = 3702

    WS_DISCOVERY_PROBE = '''<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
    xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
    xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
    xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
    <e:Header>
        <w:MessageID>uuid:{uuid}</w:MessageID>
        <w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
        <w:Action e:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
    </e:Header>
    <e:Body>
        <d:Probe>
            <d:Types>dn:NetworkVideoTransmitter</d:Types>
        </d:Probe>
    </e:Body>
</e:Envelope>'''

    @classmethod
    def discover(cls, timeout: float = 5.0) -> list:
        """Discover ONVIF cameras on the network."""
        cameras = []

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            sock.settimeout(timeout)

            # Send probe
            probe = cls.WS_DISCOVERY_PROBE.format(uuid=str(uuid.uuid4()))
            sock.sendto(probe.encode(), (cls.MULTICAST_IP, cls.MULTICAST_PORT))

            # Collect responses
            end_time = time.time() + timeout
            seen_ips = set()

            while time.time() < end_time:
                try:
                    data, addr = sock.recvfrom(65535)
                    ip = addr[0]

                    if ip in seen_ips:
                        continue
                    seen_ips.add(ip)

                    # Parse response
                    response = data.decode('utf-8', errors='ignore')
                    camera = cls._parse_response(response, ip)
                    if camera:
                        cameras.append(camera)

                except socket.timeout:
                    break
                except Exception as e:
                    logger.debug(f"Error receiving discovery response: {e}")
                    continue

            sock.close()

        except Exception as e:
            logger.error(f"Discovery error: {e}")

        return cameras

    @classmethod
    def _parse_response(cls, response: str, ip: str) -> Optional[dict]:
        """Parse WS-Discovery response."""
        try:
            # Extract XAddrs (service addresses)
            xaddrs_match = re.search(r'<[^>]*XAddrs[^>]*>([^<]+)<', response)

            # Try to extract manufacturer/model
            manufacturer = ""
            model = ""

            # Common patterns in responses
            if "Hikvision" in response or "hikvision" in response:
                manufacturer = "Hikvision"
            elif "Dahua" in response or "dahua" in response:
                manufacturer = "Dahua"
            elif "Tapo" in response or "TP-Link" in response:
                manufacturer = "TP-Link Tapo"
            elif "Reolink" in response or "reolink" in response:
                manufacturer = "Reolink"
            elif "Amcrest" in response:
                manufacturer = "Amcrest"

            # Generate RTSP URL based on manufacturer
            rtsp_url = cls._generate_rtsp_url(ip, manufacturer)

            return {
                'ip': ip,
                'port': 554,
                'manufacturer': manufacturer,
                'model': model,
                'name': f"{manufacturer or 'Camera'} @ {ip}",
                'stream_url': rtsp_url
            }

        except Exception as e:
            logger.debug(f"Error parsing response from {ip}: {e}")
            return None

    @staticmethod
    def _generate_rtsp_url(ip: str, manufacturer: str) -> str:
        """Generate RTSP URL based on manufacturer."""
        templates = {
            'Hikvision': f'rtsp://{{user}}:{{pass}}@{ip}:554/Streaming/Channels/101',
            'Dahua': f'rtsp://{{user}}:{{pass}}@{ip}:554/cam/realmonitor?channel=1&subtype=0',
            'TP-Link Tapo': f'rtsp://{{user}}:{{pass}}@{ip}:554/stream1',
            'Reolink': f'rtsp://{{user}}:{{pass}}@{ip}:554/h264Preview_01_main',
            'Amcrest': f'rtsp://{{user}}:{{pass}}@{ip}:554/cam/realmonitor?channel=1&subtype=0',
        }
        return templates.get(manufacturer, f'rtsp://{{user}}:{{pass}}@{ip}:554/stream1')


class CameraProcessor:
    """Process camera streams and extract values with template matching support."""

    def __init__(self):
        self.cameras: dict[str, CameraConfig] = {}
        self.values: dict[str, ExtractedValue] = {}
        self.template_matcher = TemplateMatcher(TEMPLATES_PATH)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def load_config(self) -> int:
        """Load configuration from options.json."""
        try:
            if not os.path.exists(OPTIONS_PATH):
                logger.warning(f"Options file not found: {OPTIONS_PATH}")
                return 30

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
                    use_template_matching=cam_config.get('use_template_matching', False),
                )
                self.cameras[camera.name] = camera
                logger.info(f"Loaded camera: {camera.name}")

            return options.get('scan_interval', 30)

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return 30

    def save_config(self, cameras_list: list = None, settings: dict = None):
        """Save configuration to options.json."""
        try:
            # Load existing config
            if os.path.exists(OPTIONS_PATH):
                with open(OPTIONS_PATH, 'r') as f:
                    options = json.load(f)
            else:
                options = {'cameras': [], 'scan_interval': 30, 'log_level': 'info'}

            # Update cameras if provided
            if cameras_list is not None:
                options['cameras'] = cameras_list

            # Update settings if provided
            if settings:
                options.update(settings)

            # Save
            with open(OPTIONS_PATH, 'w') as f:
                json.dump(options, f, indent=2)

            # Reload config
            self.load_config()

            logger.info("Configuration saved")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False

    def get_authenticated_url(self, camera: CameraConfig) -> str:
        """Build authenticated stream URL."""
        url = camera.stream_url

        if not camera.username or not camera.password:
            return url

        if "@" in url:
            return url

        if "://" in url:
            protocol, rest = url.split("://", 1)
            return f"{protocol}://{camera.username}:{camera.password}@{rest}"

        return url

    def capture_frame(self, camera: CameraConfig) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Capture a frame from camera."""
        try:
            url = self.get_authenticated_url(camera)
            cap = cv2.VideoCapture(url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                return None, "Failed to open stream"

            # Read a few frames to get latest
            frame = None
            for _ in range(3):
                ret, frame = cap.read()

            cap.release()

            if not ret or frame is None:
                return None, "Failed to read frame"

            return frame, None

        except Exception as e:
            return None, str(e)

    def capture_frame_from_url(self, url: str, username: str = "", password: str = "") -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Capture a frame from any URL."""
        try:
            full_url = url
            if username and password and "@" not in url and "://" in url:
                protocol, rest = url.split("://", 1)
                full_url = f"{protocol}://{username}:{password}@{rest}"

            cap = cv2.VideoCapture(full_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                return None, "Failed to open stream"

            frame = None
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
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
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
        """Extract numeric value from frame with optional template matching."""
        try:
            roi_x, roi_y, roi_width, roi_height = camera.roi_x, camera.roi_y, camera.roi_width, camera.roi_height

            # Use template matching if enabled
            if camera.use_template_matching and camera.template_name:
                match = self.template_matcher.find_template(frame, camera.template_name)
                if match and match['confidence'] > 0.6:
                    roi_x = match['x']
                    roi_y = match['y']
                    roi_width = match['width']
                    roi_height = match['height']
                    logger.debug(f"Template matched at ({roi_x}, {roi_y}) with confidence {match['confidence']:.2f}")

            # Apply ROI if specified
            if roi_width > 0 and roi_height > 0:
                img_h, img_w = frame.shape[:2]
                x = max(0, min(roi_x, img_w - 1))
                y = max(0, min(roi_y, img_h - 1))
                w = min(roi_width, img_w - x)
                h = min(roi_height, img_h - y)
                roi_frame = frame[y:y+h, x:x+w]
            else:
                roi_frame = frame
                roi_x, roi_y = 0, 0
                roi_height, roi_width = frame.shape[:2]

            # Preprocess
            processed = self.preprocess_image(roi_frame, camera.preprocessing)

            # OCR
            config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.-"
            raw_text = pytesseract.image_to_string(processed, config=config).strip()

            # Get confidence
            data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
            confidences = [int(c) for c in data["conf"] if str(c).isdigit() and int(c) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            # Parse value
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
                value=round(value, 2) if value is not None else None,
                raw_text=raw_text,
                confidence=avg_confidence,
                timestamp=time.time(),
                roi_x=roi_x,
                roi_y=roi_y,
                roi_width=roi_width,
                roi_height=roi_height,
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

    def extract_from_frame(self, frame: np.ndarray, roi: dict, preprocessing: str = "auto",
                           template_name: str = None) -> dict:
        """Extract value from a frame with given ROI or template."""
        try:
            roi_x, roi_y, roi_width, roi_height = roi.get('x', 0), roi.get('y', 0), roi.get('width', 0), roi.get('height', 0)
            matched_roi = None

            # Use template matching if specified
            if template_name:
                match = self.template_matcher.find_template(frame, template_name)
                if match and match['confidence'] > 0.6:
                    roi_x = match['x']
                    roi_y = match['y']
                    roi_width = match['width']
                    roi_height = match['height']
                    matched_roi = match

            # Apply ROI
            if roi_width > 0 and roi_height > 0:
                img_h, img_w = frame.shape[:2]
                x = max(0, min(roi_x, img_w - 1))
                y = max(0, min(roi_y, img_h - 1))
                w = min(roi_width, img_w - x)
                h = min(roi_height, img_h - y)
                roi_frame = frame[y:y+h, x:x+w]
            else:
                roi_frame = frame

            # Preprocess
            processed = self.preprocess_image(roi_frame, preprocessing)

            # OCR
            config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.-"
            raw_text = pytesseract.image_to_string(processed, config=config).strip()

            # Get confidence
            data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
            confidences = [int(c) for c in data["conf"] if str(c).isdigit() and int(c) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            # Parse value
            value = None
            for pattern in [r"-?\d+\.\d+", r"-?\d+,\d+", r"-?\d+"]:
                match = re.search(pattern, raw_text)
                if match:
                    try:
                        value = float(match.group().replace(",", "."))
                        break
                    except ValueError:
                        continue

            # Encode processed image for preview
            _, buffer = cv2.imencode('.png', processed)
            processed_b64 = base64.b64encode(buffer).decode('utf-8')

            return {
                'success': True,
                'value': round(value, 2) if value is not None else None,
                'raw_text': raw_text,
                'confidence': avg_confidence,
                'processed_image': processed_b64,
                'roi': {
                    'x': roi_x,
                    'y': roi_y,
                    'width': roi_width,
                    'height': roi_height
                },
                'matched_roi': matched_roi
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def draw_roi_on_frame(self, frame: np.ndarray, roi: dict, value: any = None,
                          unit: str = "", confidence: float = 0) -> np.ndarray:
        """Draw ROI rectangle and value on frame."""
        result = frame.copy()

        x, y, w, h = roi.get('x', 0), roi.get('y', 0), roi.get('width', 0), roi.get('height', 0)

        if w > 0 and h > 0:
            # Draw rectangle
            color = (0, 255, 0) if confidence > 50 else (0, 165, 255) if confidence > 30 else (0, 0, 255)
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

            # Draw value label
            if value is not None:
                label = f"{value}{unit}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2

                # Get text size
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Draw background
                label_y = y - 10 if y > 30 else y + h + 25
                cv2.rectangle(result, (x, label_y - text_h - 5), (x + text_w + 10, label_y + 5), color, -1)

                # Draw text
                cv2.putText(result, label, (x + 5, label_y), font, font_scale, (255, 255, 255), thickness)

                # Draw confidence
                conf_label = f"{confidence:.0f}%"
                cv2.putText(result, conf_label, (x + w + 5, y + 15), font, 0.5, color, 1)

        return result

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
            for camera in list(self.cameras.values()):
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


# ============================================================================
# Web UI Template
# ============================================================================

WEB_UI = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Camera Data Extractor</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        :root {
            --primary: #03a9f4;
            --primary-dark: #0288d1;
            --success: #4caf50;
            --warning: #ff9800;
            --error: #f44336;
            --bg-1: #0d1117;
            --bg-2: #161b22;
            --bg-3: #21262d;
            --bg-4: #30363d;
            --border: #30363d;
            --text-1: #f0f6fc;
            --text-2: #8b949e;
            --text-3: #6e7681;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-1);
            color: var(--text-1);
            min-height: 100vh;
        }

        /* Layout */
        .app { display: flex; flex-direction: column; min-height: 100vh; }

        .header {
            background: var(--bg-2);
            border-bottom: 1px solid var(--border);
            padding: 12px 20px;
            display: flex;
            align-items: center;
            gap: 20px;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 18px;
            font-weight: 600;
        }

        .logo svg { width: 24px; height: 24px; color: var(--primary); }

        .nav {
            display: flex;
            gap: 4px;
        }

        .nav-btn {
            padding: 8px 16px;
            border: none;
            background: transparent;
            color: var(--text-2);
            cursor: pointer;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.15s;
        }

        .nav-btn:hover { background: var(--bg-3); color: var(--text-1); }
        .nav-btn.active { background: var(--primary); color: white; }

        .main { flex: 1; padding: 20px; max-width: 1600px; margin: 0 auto; width: 100%; }

        .page { display: none; }
        .page.active { display: block; animation: fadeIn 0.2s; }

        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

        /* Cards */
        .card {
            background: var(--bg-2);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 16px;
        }

        .card-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Grid */
        .grid { display: grid; gap: 16px; }
        .grid-2 { grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
        .grid-3 { grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); }

        /* Dashboard Cards */
        .value-card {
            background: var(--bg-2);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            position: relative;
            overflow: hidden;
        }

        .value-card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
        }

        .value-card-name { font-size: 14px; color: var(--text-2); font-weight: 500; }
        .value-card-status { font-size: 11px; padding: 2px 8px; border-radius: 10px; }
        .value-card-status.ok { background: rgba(76, 175, 80, 0.2); color: var(--success); }
        .value-card-status.error { background: rgba(244, 67, 54, 0.2); color: var(--error); }

        .value-card-value {
            font-size: 42px;
            font-weight: 700;
            line-height: 1;
            margin-bottom: 8px;
        }

        .value-card-unit { font-size: 18px; color: var(--text-2); margin-left: 4px; }
        .value-card-meta { font-size: 12px; color: var(--text-3); }

        /* Buttons */
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .btn-primary { background: var(--primary); color: white; }
        .btn-primary:hover { background: var(--primary-dark); }
        .btn-secondary { background: var(--bg-3); color: var(--text-1); }
        .btn-secondary:hover { background: var(--bg-4); }
        .btn-success { background: var(--success); color: white; }
        .btn-danger { background: var(--error); color: white; }
        .btn-sm { padding: 6px 12px; font-size: 13px; }
        .btn-icon { padding: 8px; }

        .btn svg { width: 16px; height: 16px; }

        /* Forms */
        .form-group { margin-bottom: 16px; }
        .form-label { display: block; font-size: 13px; font-weight: 500; margin-bottom: 6px; color: var(--text-2); }

        .form-input, .form-select {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--bg-3);
            color: var(--text-1);
            font-size: 14px;
        }

        .form-input:focus, .form-select:focus {
            outline: none;
            border-color: var(--primary);
        }

        .form-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
        }

        .form-check {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .form-check input { width: 16px; height: 16px; }

        /* Preview Container */
        .preview-container {
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 20px;
            min-height: 500px;
        }

        @media (max-width: 1000px) {
            .preview-container { grid-template-columns: 1fr; }
        }

        .preview-frame {
            background: var(--bg-3);
            border-radius: 8px;
            overflow: hidden;
            position: relative;
        }

        .preview-frame img {
            width: 100%;
            height: auto;
            display: block;
        }

        .preview-canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            cursor: crosshair;
        }

        .preview-placeholder {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 400px;
            color: var(--text-3);
        }

        .preview-placeholder svg { width: 64px; height: 64px; margin-bottom: 16px; opacity: 0.5; }

        .preview-sidebar { display: flex; flex-direction: column; gap: 16px; }

        /* ROI Display */
        .roi-display {
            background: var(--bg-3);
            border-radius: 6px;
            padding: 12px;
        }

        .roi-values {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
        }

        .roi-value {
            background: var(--bg-2);
            padding: 8px;
            border-radius: 4px;
            text-align: center;
        }

        .roi-value-label { font-size: 11px; color: var(--text-3); text-transform: uppercase; }
        .roi-value-num { font-size: 18px; font-weight: 600; font-family: monospace; }

        /* Result Display */
        .result-display {
            background: var(--bg-3);
            border-radius: 6px;
            padding: 16px;
            text-align: center;
        }

        .result-value { font-size: 48px; font-weight: 700; color: var(--success); }
        .result-raw { font-size: 13px; color: var(--text-3); margin-top: 8px; }
        .result-confidence { font-size: 12px; color: var(--text-2); margin-top: 4px; }

        /* Camera List */
        .camera-item {
            background: var(--bg-3);
            border-radius: 8px;
            padding: 16px;
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .camera-item-preview {
            width: 120px;
            height: 80px;
            background: var(--bg-2);
            border-radius: 6px;
            overflow: hidden;
            flex-shrink: 0;
        }

        .camera-item-preview img { width: 100%; height: 100%; object-fit: cover; }

        .camera-item-info { flex: 1; min-width: 0; }
        .camera-item-name { font-weight: 600; margin-bottom: 4px; }
        .camera-item-url { font-size: 13px; color: var(--text-3); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .camera-item-actions { display: flex; gap: 8px; }

        /* Discovery */
        .discovery-item {
            background: var(--bg-3);
            border-radius: 8px;
            padding: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .discovery-item-info h4 { font-size: 15px; margin-bottom: 4px; }
        .discovery-item-info p { font-size: 13px; color: var(--text-3); }

        /* Template List */
        .template-item {
            background: var(--bg-3);
            border-radius: 8px;
            padding: 12px;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .template-preview {
            width: 80px;
            height: 60px;
            background: var(--bg-2);
            border-radius: 4px;
            overflow: hidden;
        }

        .template-preview img { width: 100%; height: 100%; object-fit: contain; }
        .template-info { flex: 1; }
        .template-name { font-weight: 500; margin-bottom: 2px; }
        .template-meta { font-size: 12px; color: var(--text-3); }

        /* Modal */
        .modal {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.7);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }

        .modal.active { display: flex; }

        .modal-content {
            background: var(--bg-2);
            border: 1px solid var(--border);
            border-radius: 12px;
            width: 90%;
            max-width: 500px;
            max-height: 90vh;
            overflow-y: auto;
        }

        .modal-header {
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .modal-title { font-size: 18px; font-weight: 600; }
        .modal-close { background: none; border: none; color: var(--text-2); cursor: pointer; padding: 4px; }
        .modal-close:hover { color: var(--text-1); }
        .modal-body { padding: 20px; }
        .modal-footer { padding: 16px 20px; border-top: 1px solid var(--border); display: flex; justify-content: flex-end; gap: 8px; }

        /* Toast */
        .toast-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 2000;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .toast {
            background: var(--bg-3);
            border: 1px solid var(--border);
            padding: 12px 16px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            gap: 10px;
            animation: slideIn 0.2s;
            min-width: 250px;
        }

        @keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }

        .toast.success { border-left: 3px solid var(--success); }
        .toast.error { border-left: 3px solid var(--error); }
        .toast.info { border-left: 3px solid var(--primary); }

        /* Live indicator */
        .live-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(244, 67, 54, 0.2);
            color: var(--error);
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }

        .live-badge::before {
            content: '';
            width: 8px;
            height: 8px;
            background: var(--error);
            border-radius: 50%;
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-3);
        }

        .empty-state svg { width: 64px; height: 64px; margin-bottom: 16px; opacity: 0.5; }
        .empty-state h3 { font-size: 18px; color: var(--text-2); margin-bottom: 8px; }
        .empty-state p { font-size: 14px; margin-bottom: 20px; }

        /* Loading */
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid var(--bg-4);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        /* Status bar */
        .status-bar {
            background: var(--bg-2);
            border-top: 1px solid var(--border);
            padding: 8px 20px;
            font-size: 12px;
            color: var(--text-3);
            display: flex;
            align-items: center;
            gap: 20px;
        }

        .status-item { display: flex; align-items: center; gap: 6px; }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; }
        .status-dot.online { background: var(--success); }
        .status-dot.offline { background: var(--error); }
    </style>
</head>
<body>
    <div class="app">
        <header class="header">
            <div class="logo">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                    <circle cx="12" cy="13" r="4"></circle>
                </svg>
                Camera Data Extractor
            </div>
            <nav class="nav">
                <button class="nav-btn active" data-page="dashboard">Dashboard</button>
                <button class="nav-btn" data-page="cameras">Cameras</button>
                <button class="nav-btn" data-page="live">Live Preview</button>
                <button class="nav-btn" data-page="templates">Templates</button>
                <button class="nav-btn" data-page="discover">Discover</button>
            </nav>
        </header>

        <main class="main">
            <!-- Dashboard Page -->
            <div id="page-dashboard" class="page active">
                <div class="card">
                    <div class="card-title">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
                            <rect x="3" y="3" width="7" height="7"></rect>
                            <rect x="14" y="3" width="7" height="7"></rect>
                            <rect x="14" y="14" width="7" height="7"></rect>
                            <rect x="3" y="14" width="7" height="7"></rect>
                        </svg>
                        Extracted Values
                        <span class="live-badge">LIVE</span>
                    </div>
                    <div id="values-grid" class="grid grid-3">
                        <div class="empty-state">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                                <circle cx="12" cy="13" r="4"></circle>
                            </svg>
                            <h3>No cameras configured</h3>
                            <p>Add cameras to start extracting values</p>
                            <button class="btn btn-primary" onclick="showPage('cameras')">Add Camera</button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Cameras Page -->
            <div id="page-cameras" class="page">
                <div class="card">
                    <div class="card-title" style="justify-content: space-between;">
                        <span>Configured Cameras</span>
                        <button class="btn btn-primary" onclick="openAddCameraModal()">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
                                <line x1="12" y1="5" x2="12" y2="19"></line>
                                <line x1="5" y1="12" x2="19" y2="12"></line>
                            </svg>
                            Add Camera
                        </button>
                    </div>
                    <div id="cameras-list" class="grid"></div>
                </div>
            </div>

            <!-- Live Preview Page -->
            <div id="page-live" class="page">
                <div class="card">
                    <div class="card-title">Live Preview & ROI Selection</div>
                    <div class="form-row" style="margin-bottom: 16px;">
                        <div class="form-group" style="flex: 2;">
                            <label class="form-label">Select Camera</label>
                            <select id="live-camera-select" class="form-select" onchange="loadLivePreview()">
                                <option value="">-- Select a camera --</option>
                            </select>
                        </div>
                        <div class="form-group" style="flex: 1;">
                            <label class="form-label">Preprocessing</label>
                            <select id="live-preprocessing" class="form-select">
                                <option value="auto">Auto</option>
                                <option value="none">None</option>
                                <option value="threshold">Threshold</option>
                                <option value="adaptive">Adaptive</option>
                                <option value="invert">Invert</option>
                            </select>
                        </div>
                        <div class="form-group" style="display: flex; align-items: flex-end;">
                            <button class="btn btn-primary" onclick="refreshLivePreview()">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
                                    <polyline points="23 4 23 10 17 10"></polyline>
                                    <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
                                </svg>
                                Refresh
                            </button>
                        </div>
                    </div>

                    <div class="preview-container">
                        <div class="preview-frame" id="preview-frame">
                            <div class="preview-placeholder" id="preview-placeholder">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                                    <circle cx="12" cy="13" r="4"></circle>
                                </svg>
                                <p>Select a camera and click Refresh to load preview</p>
                            </div>
                            <img id="preview-image" style="display: none;" />
                            <canvas id="preview-canvas" class="preview-canvas" style="display: none;"></canvas>
                        </div>

                        <div class="preview-sidebar">
                            <div class="card" style="margin: 0;">
                                <div class="card-title">Region of Interest (ROI)</div>
                                <p style="font-size: 13px; color: var(--text-3); margin-bottom: 12px;">
                                    Click and drag on the image to select the area containing the value you want to extract.
                                </p>
                                <div class="roi-display">
                                    <div class="roi-values">
                                        <div class="roi-value">
                                            <div class="roi-value-label">X</div>
                                            <div class="roi-value-num" id="roi-x">0</div>
                                        </div>
                                        <div class="roi-value">
                                            <div class="roi-value-label">Y</div>
                                            <div class="roi-value-num" id="roi-y">0</div>
                                        </div>
                                        <div class="roi-value">
                                            <div class="roi-value-label">Width</div>
                                            <div class="roi-value-num" id="roi-w">0</div>
                                        </div>
                                        <div class="roi-value">
                                            <div class="roi-value-label">Height</div>
                                            <div class="roi-value-num" id="roi-h">0</div>
                                        </div>
                                    </div>
                                </div>
                                <div style="margin-top: 12px; display: flex; gap: 8px;">
                                    <button class="btn btn-secondary btn-sm" onclick="clearROI()">Clear</button>
                                    <button class="btn btn-primary btn-sm" onclick="testExtraction()">Test Extract</button>
                                </div>
                            </div>

                            <div class="card" style="margin: 0;">
                                <div class="card-title">Extraction Result</div>
                                <div class="result-display" id="result-display">
                                    <div class="result-value" id="result-value">--</div>
                                    <div class="result-raw" id="result-raw">Draw ROI and click Test Extract</div>
                                    <div class="result-confidence" id="result-confidence"></div>
                                </div>
                            </div>

                            <div class="card" style="margin: 0;">
                                <div class="card-title">Save as Template</div>
                                <p style="font-size: 13px; color: var(--text-3); margin-bottom: 12px;">
                                    Save this ROI as a template to track it even if the camera moves or rotates.
                                </p>
                                <div class="form-group">
                                    <input type="text" id="template-name-input" class="form-input" placeholder="Template name">
                                </div>
                                <button class="btn btn-success btn-sm" onclick="saveTemplate()">Save Template</button>
                            </div>

                            <button class="btn btn-primary" onclick="applyROIToCamera()" style="width: 100%;">
                                Save ROI to Camera
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Templates Page -->
            <div id="page-templates" class="page">
                <div class="card">
                    <div class="card-title">Saved Templates</div>
                    <p style="font-size: 13px; color: var(--text-3); margin-bottom: 16px;">
                        Templates allow automatic tracking of the ROI even when the camera view changes, rotates, or moves.
                    </p>
                    <div id="templates-list" class="grid grid-3"></div>
                </div>
            </div>

            <!-- Discover Page -->
            <div id="page-discover" class="page">
                <div class="card">
                    <div class="card-title" style="justify-content: space-between;">
                        <span>Network Discovery (ONVIF)</span>
                        <button class="btn btn-primary" onclick="startDiscovery()" id="discover-btn">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
                                <circle cx="11" cy="11" r="8"></circle>
                                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                            </svg>
                            Scan Network
                        </button>
                    </div>
                    <div id="discovery-list" class="grid grid-2">
                        <div class="empty-state">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="11" cy="11" r="8"></circle>
                                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                            </svg>
                            <h3>No cameras discovered</h3>
                            <p>Click "Scan Network" to find ONVIF cameras</p>
                        </div>
                    </div>
                </div>
            </div>
        </main>

        <div class="status-bar">
            <div class="status-item">
                <div class="status-dot online" id="status-dot"></div>
                <span id="status-text">Connected</span>
            </div>
            <div class="status-item">
                <span id="camera-count">0 cameras</span>
            </div>
        </div>
    </div>

    <!-- Add/Edit Camera Modal -->
    <div class="modal" id="camera-modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3 class="modal-title" id="camera-modal-title">Add Camera</h3>
                <button class="modal-close" onclick="closeCameraModal()">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                </button>
            </div>
            <div class="modal-body">
                <input type="hidden" id="camera-edit-name">
                <div class="form-group">
                    <label class="form-label">Camera Name *</label>
                    <input type="text" id="camera-name" class="form-input" placeholder="e.g., Boiler Temperature">
                </div>
                <div class="form-group">
                    <label class="form-label">Stream URL *</label>
                    <input type="text" id="camera-url" class="form-input" placeholder="rtsp://user:pass@192.168.1.100:554/stream1">
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">Username</label>
                        <input type="text" id="camera-username" class="form-input">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Password</label>
                        <input type="password" id="camera-password" class="form-input">
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">Value Name</label>
                        <input type="text" id="camera-value-name" class="form-input" placeholder="Temperature" value="Value">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Unit</label>
                        <input type="text" id="camera-unit" class="form-input" placeholder="°C">
                    </div>
                </div>
                <div class="form-group">
                    <label class="form-label">Preprocessing</label>
                    <select id="camera-preprocessing" class="form-select">
                        <option value="auto">Auto (Recommended)</option>
                        <option value="none">None</option>
                        <option value="threshold">Threshold</option>
                        <option value="adaptive">Adaptive</option>
                        <option value="invert">Invert</option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="form-label">Template (for auto-tracking)</label>
                    <select id="camera-template" class="form-select">
                        <option value="">None</option>
                    </select>
                </div>
                <div class="form-group">
                    <div class="form-check">
                        <input type="checkbox" id="camera-use-template">
                        <label for="camera-use-template">Use template matching (auto-find ROI)</label>
                    </div>
                </div>
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" onclick="closeCameraModal()">Cancel</button>
                <button class="btn btn-primary" onclick="saveCamera()">Save Camera</button>
            </div>
        </div>
    </div>

    <div class="toast-container" id="toast-container"></div>

    <script>
        // State
        let cameras = {};
        let values = {};
        let templates = [];
        let currentROI = { x: 0, y: 0, width: 0, height: 0 };
        let isDrawing = false;
        let startX, startY;
        let previewImage = null;
        let imageScale = 1;

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            loadData();
            setInterval(loadValues, 5000);
            setupCanvas();
        });

        // Navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                showPage(btn.dataset.page);
            });
        });

        function showPage(page) {
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            document.querySelector(`[data-page="${page}"]`).classList.add('active');
            document.getElementById(`page-${page}`).classList.add('active');

            if (page === 'live') populateCameraSelect();
            if (page === 'templates') loadTemplates();
        }

        // Data Loading
        async function loadData() {
            await Promise.all([loadCameras(), loadValues(), loadTemplates()]);
        }

        async function loadCameras() {
            try {
                const res = await fetch('api/cameras');
                cameras = await res.json();
                renderCameras();
                updateStatus();
            } catch (e) {
                console.error('Failed to load cameras:', e);
            }
        }

        async function loadValues() {
            try {
                const res = await fetch('api/values');
                values = await res.json();
                renderValues();
            } catch (e) {
                console.error('Failed to load values:', e);
            }
        }

        async function loadTemplates() {
            try {
                const res = await fetch('api/templates');
                templates = await res.json();
                renderTemplates();
                populateTemplateSelect();
            } catch (e) {
                console.error('Failed to load templates:', e);
            }
        }

        // Render Functions
        function renderValues() {
            const grid = document.getElementById('values-grid');
            const entries = Object.entries(values);

            if (entries.length === 0) {
                grid.innerHTML = `
                    <div class="empty-state">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                            <circle cx="12" cy="13" r="4"></circle>
                        </svg>
                        <h3>No cameras configured</h3>
                        <p>Add cameras to start extracting values</p>
                        <button class="btn btn-primary" onclick="showPage('cameras')">Add Camera</button>
                    </div>
                `;
                return;
            }

            grid.innerHTML = entries.map(([name, data]) => {
                const camera = cameras[name] || {};
                const hasError = data.error;
                const statusClass = hasError ? 'error' : 'ok';
                const statusText = hasError ? 'Error' : 'OK';
                const displayValue = data.value !== null ? data.value : '--';
                const time = data.timestamp ? new Date(data.timestamp * 1000).toLocaleTimeString() : '';

                return `
                    <div class="value-card">
                        <div class="value-card-header">
                            <div class="value-card-name">${name}</div>
                            <div class="value-card-status ${statusClass}">${statusText}</div>
                        </div>
                        <div class="value-card-value">
                            ${displayValue}<span class="value-card-unit">${camera.unit || data.unit || ''}</span>
                        </div>
                        <div class="value-card-meta">
                            ${hasError ? `<span style="color: var(--error);">${data.error}</span>` :
                              `Confidence: ${(data.confidence || 0).toFixed(0)}% • ${time}`}
                        </div>
                    </div>
                `;
            }).join('');
        }

        function renderCameras() {
            const list = document.getElementById('cameras-list');
            const entries = Object.entries(cameras);

            if (entries.length === 0) {
                list.innerHTML = `
                    <div class="empty-state" style="grid-column: 1 / -1;">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                            <circle cx="12" cy="13" r="4"></circle>
                        </svg>
                        <h3>No cameras configured</h3>
                        <p>Click "Add Camera" to get started</p>
                    </div>
                `;
                return;
            }

            list.innerHTML = entries.map(([name, cam]) => `
                <div class="camera-item">
                    <div class="camera-item-preview" id="preview-${name.replace(/\\s/g, '-')}">
                        <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-3);">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="24" height="24">
                                <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                                <circle cx="12" cy="13" r="4"></circle>
                            </svg>
                        </div>
                    </div>
                    <div class="camera-item-info">
                        <div class="camera-item-name">${name}</div>
                        <div class="camera-item-url">${cam.stream_url}</div>
                        <div style="font-size: 12px; color: var(--text-3); margin-top: 4px;">
                            ${cam.value_name}${cam.unit ? ' (' + cam.unit + ')' : ''} •
                            ROI: ${cam.roi_width > 0 ? `${cam.roi_width}x${cam.roi_height}` : 'Not set'}
                            ${cam.use_template_matching ? ' • Template: ' + cam.template_name : ''}
                        </div>
                    </div>
                    <div class="camera-item-actions">
                        <button class="btn btn-secondary btn-sm" onclick="editCamera('${name}')">Edit</button>
                        <button class="btn btn-danger btn-sm" onclick="deleteCamera('${name}')">Delete</button>
                    </div>
                </div>
            `).join('');
        }

        function renderTemplates() {
            const list = document.getElementById('templates-list');

            if (templates.length === 0) {
                list.innerHTML = `
                    <div class="empty-state" style="grid-column: 1 / -1;">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                            <line x1="3" y1="9" x2="21" y2="9"></line>
                            <line x1="9" y1="21" x2="9" y2="9"></line>
                        </svg>
                        <h3>No templates saved</h3>
                        <p>Create templates from the Live Preview page</p>
                    </div>
                `;
                return;
            }

            list.innerHTML = templates.map(t => `
                <div class="template-item">
                    <div class="template-preview">
                        <img src="api/templates/${t.name}/image" alt="${t.name}">
                    </div>
                    <div class="template-info">
                        <div class="template-name">${t.name}</div>
                        <div class="template-meta">
                            ROI: ${t.original_roi?.width || 0}x${t.original_roi?.height || 0}
                        </div>
                    </div>
                    <button class="btn btn-danger btn-sm btn-icon" onclick="deleteTemplate('${t.name}')">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
                            <polyline points="3 6 5 6 21 6"></polyline>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        </svg>
                    </button>
                </div>
            `).join('');
        }

        // Camera Modal
        function openAddCameraModal() {
            document.getElementById('camera-modal-title').textContent = 'Add Camera';
            document.getElementById('camera-edit-name').value = '';
            document.getElementById('camera-name').value = '';
            document.getElementById('camera-url').value = '';
            document.getElementById('camera-username').value = '';
            document.getElementById('camera-password').value = '';
            document.getElementById('camera-value-name').value = 'Value';
            document.getElementById('camera-unit').value = '';
            document.getElementById('camera-preprocessing').value = 'auto';
            document.getElementById('camera-template').value = '';
            document.getElementById('camera-use-template').checked = false;
            document.getElementById('camera-modal').classList.add('active');
        }

        function editCamera(name) {
            const cam = cameras[name];
            if (!cam) return;

            document.getElementById('camera-modal-title').textContent = 'Edit Camera';
            document.getElementById('camera-edit-name').value = name;
            document.getElementById('camera-name').value = cam.name;
            document.getElementById('camera-url').value = cam.stream_url;
            document.getElementById('camera-username').value = cam.username || '';
            document.getElementById('camera-password').value = cam.password || '';
            document.getElementById('camera-value-name').value = cam.value_name || 'Value';
            document.getElementById('camera-unit').value = cam.unit || '';
            document.getElementById('camera-preprocessing').value = cam.preprocessing || 'auto';
            document.getElementById('camera-template').value = cam.template_name || '';
            document.getElementById('camera-use-template').checked = cam.use_template_matching || false;
            document.getElementById('camera-modal').classList.add('active');
        }

        function closeCameraModal() {
            document.getElementById('camera-modal').classList.remove('active');
        }

        async function saveCamera() {
            const editName = document.getElementById('camera-edit-name').value;
            const name = document.getElementById('camera-name').value.trim();
            const url = document.getElementById('camera-url').value.trim();

            if (!name || !url) {
                toast('Please fill in required fields', 'error');
                return;
            }

            const data = {
                name: name,
                stream_url: url,
                username: document.getElementById('camera-username').value,
                password: document.getElementById('camera-password').value,
                value_name: document.getElementById('camera-value-name').value || 'Value',
                unit: document.getElementById('camera-unit').value,
                preprocessing: document.getElementById('camera-preprocessing').value,
                template_name: document.getElementById('camera-template').value,
                use_template_matching: document.getElementById('camera-use-template').checked,
                roi_x: cameras[editName]?.roi_x || 0,
                roi_y: cameras[editName]?.roi_y || 0,
                roi_width: cameras[editName]?.roi_width || 0,
                roi_height: cameras[editName]?.roi_height || 0,
            };

            try {
                let res;
                if (editName) {
                    res = await fetch(`api/cameras/${encodeURIComponent(editName)}`, {
                        method: 'PUT',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                } else {
                    res = await fetch('api/cameras', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                }

                if (res.ok) {
                    toast('Camera saved successfully', 'success');
                    closeCameraModal();
                    loadCameras();
                } else {
                    const err = await res.json();
                    toast(err.error || 'Failed to save camera', 'error');
                }
            } catch (e) {
                toast('Failed to save camera', 'error');
            }
        }

        async function deleteCamera(name) {
            if (!confirm(`Delete camera "${name}"?`)) return;

            try {
                const res = await fetch(`api/cameras/${encodeURIComponent(name)}`, { method: 'DELETE' });
                if (res.ok) {
                    toast('Camera deleted', 'success');
                    loadCameras();
                    loadValues();
                }
            } catch (e) {
                toast('Failed to delete camera', 'error');
            }
        }

        // Live Preview
        function populateCameraSelect() {
            const select = document.getElementById('live-camera-select');
            select.innerHTML = '<option value="">-- Select a camera --</option>' +
                Object.keys(cameras).map(name => `<option value="${name}">${name}</option>`).join('');
        }

        function populateTemplateSelect() {
            const select = document.getElementById('camera-template');
            select.innerHTML = '<option value="">None</option>' +
                templates.map(t => `<option value="${t.name}">${t.name}</option>`).join('');
        }

        async function loadLivePreview() {
            const name = document.getElementById('live-camera-select').value;
            if (!name) return;

            const cam = cameras[name];
            if (cam) {
                currentROI = {
                    x: cam.roi_x || 0,
                    y: cam.roi_y || 0,
                    width: cam.roi_width || 0,
                    height: cam.roi_height || 0
                };
                updateROIDisplay();
            }

            await refreshLivePreview();
        }

        async function refreshLivePreview() {
            const name = document.getElementById('live-camera-select').value;
            if (!name) {
                toast('Please select a camera', 'error');
                return;
            }

            const placeholder = document.getElementById('preview-placeholder');
            placeholder.innerHTML = '<div class="loading"></div><p style="margin-top: 16px;">Loading preview...</p>';

            try {
                const res = await fetch(`api/capture/${encodeURIComponent(name)}`);
                const data = await res.json();

                if (data.error) {
                    placeholder.innerHTML = `<p style="color: var(--error);">${data.error}</p>`;
                    return;
                }

                const img = document.getElementById('preview-image');
                img.src = 'data:image/png;base64,' + data.frame;
                img.style.display = 'block';
                placeholder.style.display = 'none';

                img.onload = () => {
                    previewImage = { width: data.width, height: data.height };
                    setupCanvasSize();
                    drawROI();
                };

            } catch (e) {
                placeholder.innerHTML = `<p style="color: var(--error);">Failed to load preview</p>`;
            }
        }

        // Canvas & ROI
        function setupCanvas() {
            const canvas = document.getElementById('preview-canvas');

            canvas.addEventListener('mousedown', (e) => {
                if (!previewImage) return;
                isDrawing = true;
                const rect = canvas.getBoundingClientRect();
                startX = (e.clientX - rect.left) / imageScale;
                startY = (e.clientY - rect.top) / imageScale;
            });

            canvas.addEventListener('mousemove', (e) => {
                if (!isDrawing || !previewImage) return;
                const rect = canvas.getBoundingClientRect();
                const x = (e.clientX - rect.left) / imageScale;
                const y = (e.clientY - rect.top) / imageScale;

                currentROI = {
                    x: Math.round(Math.min(startX, x)),
                    y: Math.round(Math.min(startY, y)),
                    width: Math.round(Math.abs(x - startX)),
                    height: Math.round(Math.abs(y - startY))
                };

                updateROIDisplay();
                drawROI();
            });

            canvas.addEventListener('mouseup', () => {
                isDrawing = false;
            });

            canvas.addEventListener('mouseleave', () => {
                isDrawing = false;
            });
        }

        function setupCanvasSize() {
            const canvas = document.getElementById('preview-canvas');
            const img = document.getElementById('preview-image');

            canvas.width = img.clientWidth;
            canvas.height = img.clientHeight;
            canvas.style.display = 'block';

            imageScale = img.clientWidth / previewImage.width;
        }

        function drawROI() {
            const canvas = document.getElementById('preview-canvas');
            const ctx = canvas.getContext('2d');

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (currentROI.width > 0 && currentROI.height > 0) {
                const x = currentROI.x * imageScale;
                const y = currentROI.y * imageScale;
                const w = currentROI.width * imageScale;
                const h = currentROI.height * imageScale;

                // Draw semi-transparent overlay outside ROI
                ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
                ctx.fillRect(0, 0, canvas.width, y);
                ctx.fillRect(0, y + h, canvas.width, canvas.height - y - h);
                ctx.fillRect(0, y, x, h);
                ctx.fillRect(x + w, y, canvas.width - x - w, h);

                // Draw ROI border
                ctx.strokeStyle = '#03a9f4';
                ctx.lineWidth = 2;
                ctx.strokeRect(x, y, w, h);

                // Draw corner handles
                const handleSize = 8;
                ctx.fillStyle = '#03a9f4';
                ctx.fillRect(x - handleSize/2, y - handleSize/2, handleSize, handleSize);
                ctx.fillRect(x + w - handleSize/2, y - handleSize/2, handleSize, handleSize);
                ctx.fillRect(x - handleSize/2, y + h - handleSize/2, handleSize, handleSize);
                ctx.fillRect(x + w - handleSize/2, y + h - handleSize/2, handleSize, handleSize);

                // Draw value if available
                const val = values[document.getElementById('live-camera-select').value];
                if (val && val.value !== null) {
                    const label = val.value + (cameras[val.camera_name]?.unit || '');
                    ctx.font = 'bold 16px sans-serif';
                    const textWidth = ctx.measureText(label).width;

                    ctx.fillStyle = '#03a9f4';
                    ctx.fillRect(x, y - 25, textWidth + 12, 22);

                    ctx.fillStyle = 'white';
                    ctx.fillText(label, x + 6, y - 8);
                }
            }
        }

        function updateROIDisplay() {
            document.getElementById('roi-x').textContent = currentROI.x;
            document.getElementById('roi-y').textContent = currentROI.y;
            document.getElementById('roi-w').textContent = currentROI.width;
            document.getElementById('roi-h').textContent = currentROI.height;
        }

        function clearROI() {
            currentROI = { x: 0, y: 0, width: 0, height: 0 };
            updateROIDisplay();
            drawROI();
        }

        async function testExtraction() {
            const name = document.getElementById('live-camera-select').value;
            if (!name) {
                toast('Please select a camera', 'error');
                return;
            }

            const preprocessing = document.getElementById('live-preprocessing').value;

            try {
                const res = await fetch('api/test-extraction', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        camera_name: name,
                        roi: currentROI,
                        preprocessing: preprocessing
                    })
                });

                const data = await res.json();

                if (data.success) {
                    document.getElementById('result-value').textContent =
                        data.value !== null ? data.value : '--';
                    document.getElementById('result-raw').textContent =
                        `Raw: "${data.raw_text || ''}"`;
                    document.getElementById('result-confidence').textContent =
                        `Confidence: ${(data.confidence || 0).toFixed(1)}%`;

                    document.getElementById('result-value').style.color =
                        data.confidence > 50 ? 'var(--success)' :
                        data.confidence > 30 ? 'var(--warning)' : 'var(--error)';
                } else {
                    document.getElementById('result-value').textContent = 'Error';
                    document.getElementById('result-raw').textContent = data.error || 'Unknown error';
                    document.getElementById('result-confidence').textContent = '';
                }
            } catch (e) {
                toast('Extraction failed', 'error');
            }
        }

        async function applyROIToCamera() {
            const name = document.getElementById('live-camera-select').value;
            if (!name) {
                toast('Please select a camera', 'error');
                return;
            }

            const cam = cameras[name];
            if (!cam) return;

            const data = {
                ...cam,
                roi_x: currentROI.x,
                roi_y: currentROI.y,
                roi_width: currentROI.width,
                roi_height: currentROI.height,
                preprocessing: document.getElementById('live-preprocessing').value
            };

            try {
                const res = await fetch(`api/cameras/${encodeURIComponent(name)}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                if (res.ok) {
                    toast('ROI saved to camera', 'success');
                    loadCameras();
                } else {
                    toast('Failed to save ROI', 'error');
                }
            } catch (e) {
                toast('Failed to save ROI', 'error');
            }
        }

        // Templates
        async function saveTemplate() {
            const name = document.getElementById('live-camera-select').value;
            const templateName = document.getElementById('template-name-input').value.trim();

            if (!name || !templateName) {
                toast('Please select a camera and enter a template name', 'error');
                return;
            }

            if (currentROI.width <= 0 || currentROI.height <= 0) {
                toast('Please draw an ROI first', 'error');
                return;
            }

            try {
                const res = await fetch('api/templates', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        camera_name: name,
                        template_name: templateName,
                        roi: currentROI
                    })
                });

                if (res.ok) {
                    toast('Template saved', 'success');
                    document.getElementById('template-name-input').value = '';
                    loadTemplates();
                } else {
                    const err = await res.json();
                    toast(err.error || 'Failed to save template', 'error');
                }
            } catch (e) {
                toast('Failed to save template', 'error');
            }
        }

        async function deleteTemplate(name) {
            if (!confirm(`Delete template "${name}"?`)) return;

            try {
                const res = await fetch(`api/templates/${encodeURIComponent(name)}`, { method: 'DELETE' });
                if (res.ok) {
                    toast('Template deleted', 'success');
                    loadTemplates();
                }
            } catch (e) {
                toast('Failed to delete template', 'error');
            }
        }

        // Discovery
        async function startDiscovery() {
            const btn = document.getElementById('discover-btn');
            btn.disabled = true;
            btn.innerHTML = '<div class="loading"></div> Scanning...';

            const list = document.getElementById('discovery-list');
            list.innerHTML = '<div class="empty-state"><div class="loading"></div><p style="margin-top: 16px;">Scanning network for cameras...</p></div>';

            try {
                const res = await fetch('api/discover', { method: 'POST' });
                const discovered = await res.json();

                if (discovered.length === 0) {
                    list.innerHTML = `
                        <div class="empty-state" style="grid-column: 1 / -1;">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="11" cy="11" r="8"></circle>
                                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                            </svg>
                            <h3>No cameras found</h3>
                            <p>Make sure your cameras support ONVIF</p>
                        </div>
                    `;
                } else {
                    list.innerHTML = discovered.map(cam => `
                        <div class="discovery-item">
                            <div class="discovery-item-info">
                                <h4>${cam.name || cam.ip}</h4>
                                <p>${cam.manufacturer ? cam.manufacturer + ' • ' : ''}${cam.ip}:${cam.port}</p>
                            </div>
                            <button class="btn btn-primary btn-sm" onclick="addDiscoveredCamera('${cam.ip}', '${cam.stream_url}', '${cam.manufacturer}')">
                                Add
                            </button>
                        </div>
                    `).join('');
                }
            } catch (e) {
                list.innerHTML = '<div class="empty-state"><p style="color: var(--error);">Discovery failed</p></div>';
            }

            btn.disabled = false;
            btn.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
                    <circle cx="11" cy="11" r="8"></circle>
                    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </svg>
                Scan Network
            `;
        }

        function addDiscoveredCamera(ip, streamUrl, manufacturer) {
            document.getElementById('camera-name').value = `Camera ${ip}`;
            document.getElementById('camera-url').value = streamUrl.replace('{user}', 'admin').replace('{pass}', 'password');
            openAddCameraModal();
            document.getElementById('camera-modal-title').textContent = 'Add Discovered Camera';
        }

        // Utilities
        function updateStatus() {
            const count = Object.keys(cameras).length;
            document.getElementById('camera-count').textContent = `${count} camera${count !== 1 ? 's' : ''}`;
        }

        function toast(message, type = 'info') {
            const container = document.getElementById('toast-container');
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18">
                    ${type === 'success' ? '<polyline points="20 6 9 17 4 12"></polyline>' :
                      type === 'error' ? '<circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line>' :
                      '<circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line>'}
                </svg>
                <span>${message}</span>
            `;
            container.appendChild(toast);

            setTimeout(() => {
                toast.style.opacity = '0';
                setTimeout(() => toast.remove(), 200);
            }, 3000);
        }
    </script>
</body>
</html>
'''


# ============================================================================
# API Routes
# ============================================================================

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


@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """Get camera configurations."""
    return jsonify({name: asdict(cam) for name, cam in processor.cameras.items()})


@app.route('/api/cameras', methods=['POST'])
def add_camera():
    """Add a new camera."""
    data = request.get_json()

    if not data.get('name') or not data.get('stream_url'):
        return jsonify({'error': 'Name and stream_url are required'}), 400

    # Get current cameras
    cameras_list = [asdict(cam) for cam in processor.cameras.values()]

    # Check if name already exists
    if any(c['name'] == data['name'] for c in cameras_list):
        return jsonify({'error': 'Camera with this name already exists'}), 400

    # Add new camera
    cameras_list.append({
        'name': data['name'],
        'stream_url': data['stream_url'],
        'username': data.get('username', ''),
        'password': data.get('password', ''),
        'value_name': data.get('value_name', 'Value'),
        'unit': data.get('unit', ''),
        'roi_x': data.get('roi_x', 0),
        'roi_y': data.get('roi_y', 0),
        'roi_width': data.get('roi_width', 0),
        'roi_height': data.get('roi_height', 0),
        'preprocessing': data.get('preprocessing', 'auto'),
        'template_name': data.get('template_name', ''),
        'use_template_matching': data.get('use_template_matching', False),
    })

    if processor.save_config(cameras_list):
        return jsonify({'success': True})
    return jsonify({'error': 'Failed to save configuration'}), 500


@app.route('/api/cameras/<name>', methods=['PUT'])
def update_camera(name):
    """Update a camera."""
    data = request.get_json()

    if name not in processor.cameras:
        return jsonify({'error': 'Camera not found'}), 404

    cameras_list = []
    for cam_name, cam in processor.cameras.items():
        if cam_name == name:
            cameras_list.append({
                'name': data.get('name', cam.name),
                'stream_url': data.get('stream_url', cam.stream_url),
                'username': data.get('username', cam.username),
                'password': data.get('password', cam.password),
                'value_name': data.get('value_name', cam.value_name),
                'unit': data.get('unit', cam.unit),
                'roi_x': data.get('roi_x', cam.roi_x),
                'roi_y': data.get('roi_y', cam.roi_y),
                'roi_width': data.get('roi_width', cam.roi_width),
                'roi_height': data.get('roi_height', cam.roi_height),
                'preprocessing': data.get('preprocessing', cam.preprocessing),
                'template_name': data.get('template_name', cam.template_name),
                'use_template_matching': data.get('use_template_matching', cam.use_template_matching),
            })
        else:
            cameras_list.append(asdict(cam))

    if processor.save_config(cameras_list):
        return jsonify({'success': True})
    return jsonify({'error': 'Failed to save configuration'}), 500


@app.route('/api/cameras/<name>', methods=['DELETE'])
def delete_camera(name):
    """Delete a camera."""
    if name not in processor.cameras:
        return jsonify({'error': 'Camera not found'}), 404

    cameras_list = [asdict(cam) for cam_name, cam in processor.cameras.items() if cam_name != name]

    if processor.save_config(cameras_list):
        # Remove from values
        if name in processor.values:
            del processor.values[name]
        return jsonify({'success': True})
    return jsonify({'error': 'Failed to save configuration'}), 500


@app.route('/api/capture/<camera_name>')
def capture_frame_api(camera_name):
    """Capture a frame from a camera."""
    camera = processor.cameras.get(camera_name)
    if not camera:
        return jsonify({'error': 'Camera not found'}), 404

    frame, error = processor.capture_frame(camera)
    if error:
        return jsonify({'error': error}), 500

    # Draw ROI if configured
    if camera.roi_width > 0 and camera.roi_height > 0:
        val = processor.values.get(camera_name)
        frame = processor.draw_roi_on_frame(
            frame,
            {'x': camera.roi_x, 'y': camera.roi_y, 'width': camera.roi_width, 'height': camera.roi_height},
            val.value if val else None,
            camera.unit,
            val.confidence if val else 0
        )

    _, buffer = cv2.imencode('.png', frame)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')

    h, w = frame.shape[:2]
    return jsonify({
        'success': True,
        'frame': frame_b64,
        'width': w,
        'height': h,
    })


@app.route('/api/test-extraction', methods=['POST'])
def test_extraction():
    """Test extraction with given ROI."""
    data = request.get_json()
    camera_name = data.get('camera_name')
    roi = data.get('roi', {})
    preprocessing = data.get('preprocessing', 'auto')
    template_name = data.get('template_name')

    camera = processor.cameras.get(camera_name)
    if not camera:
        return jsonify({'success': False, 'error': 'Camera not found'}), 404

    frame, error = processor.capture_frame(camera)
    if error:
        return jsonify({'success': False, 'error': error})

    result = processor.extract_from_frame(frame, roi, preprocessing, template_name)
    return jsonify(result)


@app.route('/api/templates', methods=['GET'])
def get_templates():
    """Get all templates."""
    return jsonify(processor.template_matcher.get_template_list())


@app.route('/api/templates', methods=['POST'])
def save_template():
    """Save a new template."""
    data = request.get_json()
    camera_name = data.get('camera_name')
    template_name = data.get('template_name')
    roi = data.get('roi', {})

    if not camera_name or not template_name:
        return jsonify({'error': 'camera_name and template_name are required'}), 400

    camera = processor.cameras.get(camera_name)
    if not camera:
        return jsonify({'error': 'Camera not found'}), 404

    frame, error = processor.capture_frame(camera)
    if error:
        return jsonify({'error': error}), 500

    if processor.template_matcher.save_template(template_name, frame, roi):
        return jsonify({'success': True})
    return jsonify({'error': 'Failed to save template'}), 500


@app.route('/api/templates/<name>', methods=['DELETE'])
def delete_template(name):
    """Delete a template."""
    if processor.template_matcher.delete_template(name):
        return jsonify({'success': True})
    return jsonify({'error': 'Failed to delete template'}), 500


@app.route('/api/templates/<name>/image')
def get_template_image(name):
    """Get template image."""
    img_path = Path(TEMPLATES_PATH) / f"{name}.png"
    if not img_path.exists():
        return jsonify({'error': 'Template not found'}), 404

    with open(img_path, 'rb') as f:
        return Response(f.read(), mimetype='image/png')


@app.route('/api/discover', methods=['POST'])
def discover_cameras():
    """Discover ONVIF cameras on the network."""
    cameras = ONVIFDiscovery.discover(timeout=5.0)
    return jsonify(cameras)


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
