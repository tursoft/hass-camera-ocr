#!/usr/bin/env python3
"""Camera OCR Add-on Server with Full Admin Interface and Template Matching."""

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

# Paths - Use /config for persistent storage across reinstalls
OPTIONS_PATH = os.environ.get('OPTIONS_PATH', '/data/options.json')
DATA_PATH = os.environ.get('DATA_PATH', '/data')
CONFIG_PATH = os.environ.get('CONFIG_PATH', '/config/hass_camera_ocr')
CAMERAS_PATH = os.path.join(CONFIG_PATH, 'cameras.json')
TEMPLATES_PATH = os.path.join(CONFIG_PATH, 'templates')
HISTORY_PATH = os.path.join(CONFIG_PATH, 'history.json')
SAVED_ROIS_PATH = os.path.join(CONFIG_PATH, 'saved_rois')
HISTORY_IMAGES_PATH = os.path.join(CONFIG_PATH, 'history_images')

# Ensure directories exist
Path(CONFIG_PATH).mkdir(parents=True, exist_ok=True)
Path(TEMPLATES_PATH).mkdir(parents=True, exist_ok=True)
Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
Path(SAVED_ROIS_PATH).mkdir(parents=True, exist_ok=True)
Path(HISTORY_IMAGES_PATH).mkdir(parents=True, exist_ok=True)


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
    min_value: Optional[float] = None  # Expected minimum value (for filtering bad OCR)
    max_value: Optional[float] = None  # Expected maximum value (for filtering bad OCR)


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
    video_description: str = ""  # AI-generated scene description


@dataclass
class DiscoveredCamera:
    """Discovered camera from network scan."""
    ip: str
    port: int
    manufacturer: str = ""
    model: str = ""
    name: str = ""
    stream_url: str = ""


@dataclass
class AIProviderConfig:
    """AI provider configuration."""
    provider: str = "none"  # none, openai, anthropic, google, ollama, custom, google-vision, azure-ocr, aws-textract
    api_key: str = ""
    api_url: str = ""  # For Ollama, custom endpoints, or cloud OCR endpoints
    model: str = ""  # Model name
    region: str = ""  # AWS region for Textract
    enabled_for_ocr: bool = False
    enabled_for_description: bool = False


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
    ONVIF_PORTS = [80, 8080, 8000, 8899, 2020, 5000]  # Common ONVIF ports

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

    # ONVIF GetDeviceInformation for direct probe
    DEVICE_INFO_REQUEST = '''<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
    <s:Body xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">
        <GetDeviceInformation xmlns="http://www.onvif.org/ver10/device/wsdl"/>
    </s:Body>
</s:Envelope>'''

    @classmethod
    def discover(cls, timeout: float = 5.0) -> list:
        """Discover ONVIF cameras on the network using WS-Discovery."""
        cameras = []

        try:
            # Get local IP to determine network
            local_ip = cls._get_local_ip()
            logger.info(f"Local IP: {local_ip}")

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 4)
            sock.settimeout(1.0)  # Short timeout for each recv

            # Bind to local IP if available
            if local_ip:
                try:
                    sock.bind((local_ip, 0))
                except:
                    sock.bind(('', 0))
            else:
                sock.bind(('', 0))

            # Send probe multiple times for better reliability
            probe = cls.WS_DISCOVERY_PROBE.format(uuid=str(uuid.uuid4()))
            for _ in range(3):
                sock.sendto(probe.encode(), (cls.MULTICAST_IP, cls.MULTICAST_PORT))
                time.sleep(0.1)

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
                    logger.debug(f"WS-Discovery response from {ip}")
                    camera = cls._parse_response(response, ip)
                    if camera:
                        cameras.append(camera)

                except socket.timeout:
                    continue
                except Exception as e:
                    logger.debug(f"Error receiving discovery response: {e}")
                    continue

            sock.close()

        except Exception as e:
            logger.error(f"WS-Discovery error: {e}")

        logger.info(f"WS-Discovery found {len(cameras)} cameras")
        return cameras

    @classmethod
    def probe_direct(cls, ips_to_probe: list, timeout: float = 2.0) -> list:
        """Directly probe IPs for ONVIF devices by trying GetDeviceInformation."""
        cameras = []
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def probe_ip(ip):
            for port in cls.ONVIF_PORTS:
                try:
                    url = f"http://{ip}:{port}/onvif/device_service"
                    response = requests.post(
                        url,
                        data=cls.DEVICE_INFO_REQUEST,
                        headers={'Content-Type': 'application/soap+xml; charset=utf-8'},
                        timeout=timeout,
                        auth=None
                    )
                    if response.status_code in [200, 401, 403]:
                        # 401/403 means ONVIF is there but needs auth
                        logger.info(f"ONVIF device found at {ip}:{port}")
                        return cls._parse_device_info(response.text, ip, port)
                except requests.exceptions.RequestException:
                    continue
                except Exception as e:
                    logger.debug(f"Probe error {ip}:{port}: {e}")
                    continue
            return None

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(probe_ip, ip): ip for ip in ips_to_probe}
            for future in as_completed(futures, timeout=30):
                try:
                    result = future.result()
                    if result:
                        cameras.append(result)
                except:
                    pass

        logger.info(f"Direct probe found {len(cameras)} ONVIF cameras")
        return cameras

    @classmethod
    def _parse_device_info(cls, response: str, ip: str, port: int) -> Optional[dict]:
        """Parse ONVIF GetDeviceInformation response."""
        manufacturer = ""
        model = ""

        # Try to extract manufacturer
        mfr_match = re.search(r'<[^>]*Manufacturer[^>]*>([^<]+)<', response)
        if mfr_match:
            manufacturer = mfr_match.group(1).strip()

        # Try to extract model
        model_match = re.search(r'<[^>]*Model[^>]*>([^<]+)<', response)
        if model_match:
            model = model_match.group(1).strip()

        # Detect by common patterns if not found
        if not manufacturer:
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

        rtsp_url = cls._generate_rtsp_url(ip, manufacturer)

        return {
            'ip': ip,
            'port': 554,
            'onvif_port': port,
            'manufacturer': manufacturer,
            'model': model,
            'name': f"{manufacturer or 'Camera'} @ {ip}" + (f" ({model})" if model else ""),
            'stream_url': rtsp_url
        }

    @classmethod
    def _get_local_ip(cls) -> Optional[str]:
        """Get the local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except:
            return None

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


class PortScanner:
    """Scan network for cameras by checking common RTSP/HTTP ports."""

    # Common camera ports
    RTSP_PORTS = [554, 8554, 10554]
    HTTP_PORTS = [80, 8080, 8000, 8888, 443, 8443]
    CAMERA_PORTS = [2020, 1935]  # 2020 is Tapo, 1935 is RTMP

    # Camera detection keywords in HTTP responses
    CAMERA_KEYWORDS = ['camera', 'ipcam', 'nvr', 'dvr', 'hikvision', 'dahua', 'tapo',
                       'reolink', 'amcrest', 'foscam', 'onvif', 'rtsp', 'streaming',
                       'video', 'webcam', 'tp-link', 'tplink', 'surveillance']

    @classmethod
    def scan_network(cls, timeout: float = 0.5) -> list:
        """Scan local network for cameras by port scanning and HTTP detection."""
        cameras = []

        # Get local network range
        try:
            # Get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()

            # Calculate network range (assume /24)
            ip_parts = local_ip.split('.')
            base_ip = '.'.join(ip_parts[:3])
            logger.info(f"Scanning network {base_ip}.0/24 for cameras...")

            all_ports = cls.RTSP_PORTS + cls.HTTP_PORTS + cls.CAMERA_PORTS

            # Scan common IP ranges for cameras
            def check_ip(ip):
                open_ports = []
                for port in all_ports:
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(timeout)
                        result = sock.connect_ex((ip, port))
                        sock.close()

                        if result == 0:
                            open_ports.append(port)
                    except:
                        pass
                return {'ip': ip, 'ports': open_ports} if open_ports else None

            # Scan in parallel using threads
            from concurrent.futures import ThreadPoolExecutor, as_completed

            ips_to_scan = [f"{base_ip}.{i}" for i in range(1, 255)]
            found = []

            with ThreadPoolExecutor(max_workers=100) as executor:
                futures = {executor.submit(check_ip, ip): ip for ip in ips_to_scan}
                for future in as_completed(futures, timeout=60):
                    try:
                        result = future.result()
                        if result and result['ports']:
                            found.append(result)
                            logger.debug(f"Found open ports on {result['ip']}: {result['ports']}")
                    except:
                        pass

            logger.info(f"Port scan found {len(found)} devices with open ports")

            # Analyze found devices to identify cameras
            def identify_camera(item):
                ip = item['ip']
                ports = item['ports']

                # Check if any RTSP port is open - likely a camera
                rtsp_port = next((p for p in ports if p in cls.RTSP_PORTS), None)

                # Check HTTP for camera indicators
                http_port = next((p for p in ports if p in cls.HTTP_PORTS), None)
                manufacturer = ""
                is_camera = rtsp_port is not None

                if http_port and not is_camera:
                    # Try to detect camera by HTTP response
                    try:
                        url = f"http://{ip}:{http_port}/"
                        response = requests.get(url, timeout=2, allow_redirects=True)
                        content = response.text.lower()
                        headers = str(response.headers).lower()

                        # Check for camera keywords
                        for keyword in cls.CAMERA_KEYWORDS:
                            if keyword in content or keyword in headers:
                                is_camera = True
                                break

                        # Detect manufacturer
                        if 'tapo' in content or 'tp-link' in content or 'tplink' in headers:
                            manufacturer = "TP-Link Tapo"
                            is_camera = True
                        elif 'hikvision' in content or 'hikvision' in headers:
                            manufacturer = "Hikvision"
                            is_camera = True
                        elif 'dahua' in content or 'dahua' in headers:
                            manufacturer = "Dahua"
                            is_camera = True
                        elif 'reolink' in content or 'reolink' in headers:
                            manufacturer = "Reolink"
                            is_camera = True
                        elif 'amcrest' in content or 'amcrest' in headers:
                            manufacturer = "Amcrest"
                            is_camera = True
                        elif 'foscam' in content:
                            manufacturer = "Foscam"
                            is_camera = True

                    except:
                        pass

                # Also check for Tapo-specific port 2020
                if 2020 in ports:
                    manufacturer = "TP-Link Tapo"
                    is_camera = True

                if is_camera:
                    return {
                        'ip': ip,
                        'ports': ports,
                        'rtsp_port': rtsp_port or 554,
                        'http_port': http_port,
                        'manufacturer': manufacturer
                    }
                return None

            # Identify cameras in parallel
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = {executor.submit(identify_camera, item): item for item in found}
                for future in as_completed(futures, timeout=30):
                    try:
                        result = future.result()
                        if result:
                            ip = result['ip']
                            if ip == local_ip:
                                continue

                            manufacturer = result['manufacturer']
                            rtsp_port = result['rtsp_port']
                            rtsp_url = ONVIFDiscovery._generate_rtsp_url(ip, manufacturer)

                            cameras.append({
                                'ip': ip,
                                'port': rtsp_port,
                                'http_port': result.get('http_port'),
                                'manufacturer': manufacturer or 'Unknown',
                                'model': '',
                                'name': f"{manufacturer or 'Camera'} @ {ip}",
                                'stream_url': rtsp_url
                            })
                            logger.info(f"Identified camera at {ip} ({manufacturer or 'Unknown'})")
                    except:
                        pass

            logger.info(f"Port scan identified {len(cameras)} cameras")

        except Exception as e:
            logger.error(f"Port scan error: {e}")

        return cameras


class PTZController:
    """ONVIF PTZ control for cameras."""

    # PTZ SOAP templates
    PTZ_CONTINUOUS_MOVE = '''<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
    <s:Body>
        <ContinuousMove xmlns="http://www.onvif.org/ver20/ptz/wsdl">
            <ProfileToken>{profile}</ProfileToken>
            <Velocity>
                <PanTilt xmlns="http://www.onvif.org/ver10/schema" x="{pan}" y="{tilt}"/>
            </Velocity>
        </ContinuousMove>
    </s:Body>
</s:Envelope>'''

    PTZ_STOP = '''<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
    <s:Body>
        <Stop xmlns="http://www.onvif.org/ver20/ptz/wsdl">
            <ProfileToken>{profile}</ProfileToken>
            <PanTilt>true</PanTilt>
            <Zoom>true</Zoom>
        </Stop>
    </s:Body>
</s:Envelope>'''

    PTZ_GOTO_HOME = '''<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
    <s:Body>
        <GotoHomePosition xmlns="http://www.onvif.org/ver20/ptz/wsdl">
            <ProfileToken>{profile}</ProfileToken>
        </GotoHomePosition>
    </s:Body>
</s:Envelope>'''

    # Common profile token names used by different camera manufacturers
    PROFILE_TOKENS = ['Profile_1', 'profile_1', 'MainStream', 'Profile1', '000', '001', 'token']

    @classmethod
    def move(cls, camera_config: 'CameraConfig', direction: str, speed: float = 0.5, duration: float = 0.3) -> dict:
        """Send PTZ command to camera."""
        try:
            # Parse stream URL to get camera IP and credentials
            parsed = urlparse(camera_config.stream_url)
            host = parsed.hostname
            username = camera_config.username or parsed.username or ''
            password = camera_config.password or parsed.password or ''

            if not host:
                return {'error': 'Could not determine camera IP from stream URL'}

            # Try different ONVIF PTZ service URLs
            ptz_urls = [
                f"http://{host}/onvif/PTZ",
                f"http://{host}:80/onvif/PTZ",
                f"http://{host}/onvif/ptz_service",
                f"http://{host}:8080/onvif/PTZ",
            ]

            # Determine pan/tilt values based on direction
            pan = 0.0
            tilt = 0.0
            if direction == 'left':
                pan = -speed
            elif direction == 'right':
                pan = speed
            elif direction == 'up':
                tilt = speed
            elif direction == 'down':
                tilt = -speed
            elif direction == 'home':
                for ptz_url in ptz_urls:
                    for profile in cls.PROFILE_TOKENS:
                        result = cls._goto_home(ptz_url, username, password, profile)
                        if result.get('success'):
                            return result
                return {'error': 'PTZ home command failed on all endpoints'}

            # Try different URLs and profiles
            last_error = None
            for ptz_url in ptz_urls:
                for profile in cls.PROFILE_TOKENS:
                    logger.debug(f"Trying PTZ: URL={ptz_url}, profile={profile}")
                    move_result = cls._send_ptz_command(
                        ptz_url, username, password,
                        cls.PTZ_CONTINUOUS_MOVE.format(profile=profile, pan=pan, tilt=tilt)
                    )

                    if move_result.get('success'):
                        # Wait for movement
                        time.sleep(duration)

                        # Stop movement
                        cls._send_ptz_command(
                            ptz_url, username, password,
                            cls.PTZ_STOP.format(profile=profile)
                        )

                        logger.info(f"PTZ success: URL={ptz_url}, profile={profile}")
                        return {'success': True, 'direction': direction}

                    last_error = move_result.get('error')
                    # Don't try more profiles if it's a connection error
                    if 'Connection failed' in str(last_error):
                        break

            return {'error': f'PTZ command failed: {last_error}'}

        except Exception as e:
            logger.error(f"PTZ error: {e}")
            return {'error': str(e)}

    @classmethod
    def _goto_home(cls, ptz_url: str, username: str, password: str, profile: str = 'Profile_1') -> dict:
        """Go to home position."""
        return cls._send_ptz_command(
            ptz_url, username, password,
            cls.PTZ_GOTO_HOME.format(profile=profile)
        )

    @classmethod
    def _send_ptz_command(cls, url: str, username: str, password: str, soap_body: str) -> dict:
        """Send ONVIF PTZ SOAP command."""
        import urllib.request
        import urllib.error

        try:
            headers = {
                'Content-Type': 'application/soap+xml; charset=utf-8',
            }

            req = urllib.request.Request(url, data=soap_body.encode('utf-8'), headers=headers, method='POST')

            # Add basic auth if credentials provided
            if username and password:
                import base64
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                req.add_header('Authorization', f'Basic {credentials}')

            with urllib.request.urlopen(req, timeout=5) as response:
                response.read()
                return {'success': True}

        except urllib.error.HTTPError as e:
            logger.warning(f"PTZ HTTP error: {e.code} - {e.reason}")
            # Try digest auth fallback or return error
            if e.code == 401:
                return {'error': 'Authentication failed. Check camera credentials.'}
            return {'error': f'HTTP {e.code}: {e.reason}'}
        except urllib.error.URLError as e:
            logger.warning(f"PTZ URL error: {e.reason}")
            return {'error': f'Connection failed: {e.reason}'}
        except Exception as e:
            logger.error(f"PTZ command error: {e}")
            return {'error': str(e)}


class AIService:
    """AI service for OCR enhancement and scene description."""

    # Storage path for AI config
    AI_CONFIG_PATH = os.path.join(CONFIG_PATH, 'ai_config.json')

    # Default models for each provider
    DEFAULT_MODELS = {
        'openai': 'gpt-4o',
        'anthropic': 'claude-sonnet-4-20250514',
        'google': 'gemini-1.5-flash',
        'ollama': 'llava',
        'google-vision': 'document-text-detection',
        'google-docai': 'ocr-processor',
        'azure-ocr': 'read',
        'aws-textract': 'detect-document-text'
    }

    _config: Optional[AIProviderConfig] = None

    @classmethod
    def load_config(cls) -> AIProviderConfig:
        """Load AI configuration."""
        if cls._config is not None:
            return cls._config

        try:
            if os.path.exists(cls.AI_CONFIG_PATH):
                with open(cls.AI_CONFIG_PATH, 'r') as f:
                    data = json.load(f)
                    cls._config = AIProviderConfig(**data)
            else:
                cls._config = AIProviderConfig()
        except Exception as e:
            logger.error(f"Error loading AI config: {e}")
            cls._config = AIProviderConfig()

        return cls._config

    @classmethod
    def save_config(cls, config: AIProviderConfig):
        """Save AI configuration."""
        cls._config = config
        try:
            with open(cls.AI_CONFIG_PATH, 'w') as f:
                json.dump(asdict(config), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving AI config: {e}")

    @classmethod
    def get_config(cls) -> dict:
        """Get AI configuration as dict (hide API key)."""
        config = cls.load_config()
        return {
            'provider': config.provider,
            'api_url': config.api_url,
            'model': config.model,
            'region': config.region,
            'enabled_for_ocr': config.enabled_for_ocr,
            'enabled_for_description': config.enabled_for_description,
            'has_api_key': bool(config.api_key)
        }

    @classmethod
    def describe_scene(cls, image_base64: str) -> str:
        """Generate scene description using AI."""
        config = cls.load_config()

        if not config.enabled_for_description or config.provider == 'none':
            return ""

        try:
            if config.provider == 'openai' or config.provider == 'custom':
                return cls._call_openai(image_base64, "scene")
            elif config.provider == 'anthropic':
                return cls._call_anthropic(image_base64, "scene")
            elif config.provider == 'google':
                return cls._call_google(image_base64, "scene")
            elif config.provider == 'ollama':
                return cls._call_ollama(image_base64, "scene")
        except Exception as e:
            logger.error(f"AI scene description error: {e}")
            return f"Error: {str(e)}"

        return ""

    @classmethod
    def enhance_ocr(cls, image_base64: str, roi_image_base64: str = None) -> Optional[str]:
        """Use AI to extract/verify OCR value."""
        config = cls.load_config()

        if not config.enabled_for_ocr or config.provider == 'none':
            return None

        image_to_use = roi_image_base64 or image_base64

        try:
            if config.provider == 'openai' or config.provider == 'custom':
                return cls._call_openai(image_to_use, "ocr")
            elif config.provider == 'anthropic':
                return cls._call_anthropic(image_to_use, "ocr")
            elif config.provider == 'google':
                return cls._call_google(image_to_use, "ocr")
            elif config.provider == 'ollama':
                return cls._call_ollama(image_to_use, "ocr")
            elif config.provider == 'google-vision':
                return cls._call_google_vision(image_to_use)
            elif config.provider == 'google-docai':
                return cls._call_google_docai(image_to_use)
            elif config.provider == 'azure-ocr':
                return cls._call_azure_ocr(image_to_use)
            elif config.provider == 'aws-textract':
                return cls._call_aws_textract(image_to_use)
        except Exception as e:
            logger.error(f"AI OCR error: {e}")

        return None

    @classmethod
    def _get_prompt(cls, task: str) -> str:
        """Get prompt for task."""
        if task == "scene":
            return "Describe this camera scene in one concise sentence. Include: location type, weather/lighting, time of day if visible, any people or activities, and notable objects. Be factual and brief."
        elif task == "ocr":
            return "Extract the numeric value shown in this image. Return ONLY the number (with decimal point if present). If no number is visible, return 'none'."
        return ""

    @classmethod
    def _call_openai(cls, image_base64: str, task: str) -> str:
        """Call OpenAI API or OpenAI-compatible custom endpoint."""
        import urllib.request

        config = cls.load_config()
        model = config.model or cls.DEFAULT_MODELS['openai']

        # Use custom URL if provider is 'custom', otherwise use OpenAI default
        if config.provider == 'custom' and config.api_url:
            api_url = config.api_url.rstrip('/')
            if not api_url.endswith('/chat/completions'):
                api_url = f'{api_url}/chat/completions'
        else:
            api_url = 'https://api.openai.com/v1/chat/completions'

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {config.api_key}'
        }

        data = {
            'model': model,
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': cls._get_prompt(task)},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_base64}'}}
                ]
            }],
            'max_tokens': 300
        }

        req = urllib.request.Request(
            api_url,
            data=json.dumps(data).encode('utf-8'),
            headers=headers,
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result['choices'][0]['message']['content'].strip()

    @classmethod
    def _call_anthropic(cls, image_base64: str, task: str) -> str:
        """Call Anthropic API."""
        import urllib.request

        config = cls.load_config()
        model = config.model or cls.DEFAULT_MODELS['anthropic']

        headers = {
            'Content-Type': 'application/json',
            'x-api-key': config.api_key,
            'anthropic-version': '2023-06-01'
        }

        data = {
            'model': model,
            'max_tokens': 300,
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': image_base64}},
                    {'type': 'text', 'text': cls._get_prompt(task)}
                ]
            }]
        }

        req = urllib.request.Request(
            'https://api.anthropic.com/v1/messages',
            data=json.dumps(data).encode('utf-8'),
            headers=headers,
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result['content'][0]['text'].strip()

    @classmethod
    def _call_google(cls, image_base64: str, task: str) -> str:
        """Call Google Gemini API."""
        import urllib.request

        config = cls.load_config()
        model = config.model or cls.DEFAULT_MODELS['google']

        url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={config.api_key}'

        data = {
            'contents': [{
                'parts': [
                    {'text': cls._get_prompt(task)},
                    {'inline_data': {'mime_type': 'image/jpeg', 'data': image_base64}}
                ]
            }]
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result['candidates'][0]['content']['parts'][0]['text'].strip()

    @classmethod
    def _call_ollama(cls, image_base64: str, task: str) -> str:
        """Call Ollama API (local)."""
        import urllib.request

        config = cls.load_config()
        model = config.model or cls.DEFAULT_MODELS['ollama']
        api_url = config.api_url or 'http://localhost:11434'

        data = {
            'model': model,
            'prompt': cls._get_prompt(task),
            'images': [image_base64],
            'stream': False
        }

        req = urllib.request.Request(
            f'{api_url}/api/generate',
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get('response', '').strip()

    @classmethod
    def _call_google_vision(cls, image_base64: str) -> str:
        """Call Google Cloud Vision API for OCR."""
        import urllib.request

        config = cls.load_config()

        url = f'https://vision.googleapis.com/v1/images:annotate?key={config.api_key}'

        data = {
            'requests': [{
                'image': {'content': image_base64},
                'features': [{'type': 'TEXT_DETECTION', 'maxResults': 10}]
            }]
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            annotations = result.get('responses', [{}])[0].get('textAnnotations', [])
            if annotations:
                # First annotation contains the full text
                full_text = annotations[0].get('description', '').strip()
                # Extract numeric value from the text
                numbers = re.findall(r'[-+]?\d*\.?\d+', full_text)
                if numbers:
                    return numbers[0]
            return 'none'

    @classmethod
    def _call_google_docai(cls, image_base64: str) -> str:
        """Call Google Document AI API for OCR."""
        import urllib.request

        config = cls.load_config()

        # Document AI requires project ID and processor ID in the URL
        # Format: api_url should be like: projects/PROJECT_ID/locations/LOCATION/processors/PROCESSOR_ID
        processor_path = config.api_url or ''
        if not processor_path:
            logger.error("Google Document AI requires processor path in API URL field")
            return 'none'

        url = f'https://documentai.googleapis.com/v1/{processor_path}:process'

        # Prepare the request
        image_data = base64.b64decode(image_base64)

        data = {
            'rawDocument': {
                'content': image_base64,
                'mimeType': 'image/jpeg'
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {config.api_key}'
            },
            method='POST'
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                # Extract text from Document AI response
                document = result.get('document', {})
                full_text = document.get('text', '').strip()
                # Extract numeric value from the text
                numbers = re.findall(r'[-+]?\d*\.?\d+', full_text)
                if numbers:
                    return numbers[0]
                return 'none'
        except Exception as e:
            logger.error(f"Google Document AI error: {e}")
            return 'none'

    @classmethod
    def _call_azure_ocr(cls, image_base64: str) -> str:
        """Call Azure Computer Vision API for OCR."""
        import urllib.request

        config = cls.load_config()
        endpoint = config.api_url or 'https://westus.api.cognitive.microsoft.com'
        endpoint = endpoint.rstrip('/')

        # First, submit the read request
        url = f'{endpoint}/vision/v3.2/read/analyze'

        image_data = base64.b64decode(image_base64)

        req = urllib.request.Request(
            url,
            data=image_data,
            headers={
                'Content-Type': 'application/octet-stream',
                'Ocp-Apim-Subscription-Key': config.api_key
            },
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            operation_location = response.headers.get('Operation-Location')

        if not operation_location:
            return 'none'

        # Poll for results
        for _ in range(10):
            time.sleep(1)
            req = urllib.request.Request(
                operation_location,
                headers={'Ocp-Apim-Subscription-Key': config.api_key},
                method='GET'
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                status = result.get('status')
                if status == 'succeeded':
                    text_lines = []
                    for read_result in result.get('analyzeResult', {}).get('readResults', []):
                        for line in read_result.get('lines', []):
                            text_lines.append(line.get('text', ''))
                    full_text = ' '.join(text_lines)
                    # Extract numeric value
                    numbers = re.findall(r'[-+]?\d*\.?\d+', full_text)
                    if numbers:
                        return numbers[0]
                    return 'none'
                elif status == 'failed':
                    return 'none'

        return 'none'

    @classmethod
    def _call_aws_textract(cls, image_base64: str) -> str:
        """Call AWS Textract API for OCR."""
        import urllib.request
        import hashlib
        import hmac
        from datetime import datetime

        config = cls.load_config()
        region = config.region or 'us-east-1'

        # AWS Textract requires signature v4 authentication
        # For simplicity, we'll use boto3 if available, otherwise basic API
        try:
            import boto3
            client = boto3.client(
                'textract',
                region_name=region,
                aws_access_key_id=config.api_key.split(':')[0] if ':' in config.api_key else config.api_key,
                aws_secret_access_key=config.api_key.split(':')[1] if ':' in config.api_key else ''
            )

            image_data = base64.b64decode(image_base64)
            response = client.detect_document_text(Document={'Bytes': image_data})

            text_lines = []
            for block in response.get('Blocks', []):
                if block['BlockType'] == 'LINE':
                    text_lines.append(block.get('Text', ''))

            full_text = ' '.join(text_lines)
            # Extract numeric value
            numbers = re.findall(r'[-+]?\d*\.?\d+', full_text)
            if numbers:
                return numbers[0]
            return 'none'
        except ImportError:
            logger.error("boto3 not installed. Install with: pip install boto3")
            return 'none'
        except Exception as e:
            logger.error(f"AWS Textract error: {e}")
            return 'none'


class HomeAssistantIntegration:
    """Integration with Home Assistant to expose sensor entities."""

    # Supervisor API endpoint
    SUPERVISOR_API = "http://supervisor/core/api"

    @classmethod
    def _get_headers(cls) -> dict:
        """Get authentication headers for Supervisor API."""
        token = os.environ.get('SUPERVISOR_TOKEN', '')
        return {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }

    @classmethod
    def _sanitize_name(cls, name: str) -> str:
        """Sanitize camera name for use in entity IDs."""
        # Convert to lowercase, replace spaces and special chars with underscores
        sanitized = re.sub(r'[^a-z0-9]', '_', name.lower())
        # Remove consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        return sanitized.strip('_')

    @classmethod
    def update_sensor(cls, camera_name: str, value: Optional[float], raw_text: str,
                      confidence: float, unit: str = "", error: str = "",
                      video_description: str = "") -> bool:
        """Update Home Assistant sensor entities for a camera."""
        import urllib.request
        import urllib.error

        token = os.environ.get('SUPERVISOR_TOKEN', '')
        if not token:
            logger.debug("SUPERVISOR_TOKEN not available, skipping HA integration")
            return False

        sanitized_name = cls._sanitize_name(camera_name)
        headers = cls._get_headers()

        try:
            # Update numeric value sensor
            value_entity_id = f"sensor.camera_ocr_{sanitized_name}_value"
            value_state = str(value) if value is not None else "unknown"
            value_data = {
                'state': value_state,
                'attributes': {
                    'friendly_name': f"{camera_name} OCR Value",
                    'unit_of_measurement': unit,
                    'device_class': 'measurement' if unit else None,
                    'confidence': round(confidence, 1) if confidence else 0,
                    'raw_text': raw_text,
                    'camera_name': camera_name,
                    'error': error,
                    'video_description': video_description,
                    'icon': 'mdi:numeric'
                }
            }

            req = urllib.request.Request(
                f"{cls.SUPERVISOR_API}/states/{value_entity_id}",
                data=json.dumps(value_data).encode('utf-8'),
                headers=headers,
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                response.read()

            # Update text sensor (raw OCR text)
            text_entity_id = f"sensor.camera_ocr_{sanitized_name}_text"
            text_data = {
                'state': raw_text[:255] if raw_text else "unknown",  # HA state max 255 chars
                'attributes': {
                    'friendly_name': f"{camera_name} OCR Text",
                    'camera_name': camera_name,
                    'full_text': raw_text,
                    'numeric_value': value,
                    'confidence': round(confidence, 1) if confidence else 0,
                    'icon': 'mdi:text-recognition'
                }
            }

            req = urllib.request.Request(
                f"{cls.SUPERVISOR_API}/states/{text_entity_id}",
                data=json.dumps(text_data).encode('utf-8'),
                headers=headers,
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                response.read()

            # Update confidence sensor
            conf_entity_id = f"sensor.camera_ocr_{sanitized_name}_confidence"
            conf_data = {
                'state': str(round(confidence, 1)) if confidence else "0",
                'attributes': {
                    'friendly_name': f"{camera_name} OCR Confidence",
                    'unit_of_measurement': '%',
                    'camera_name': camera_name,
                    'icon': 'mdi:percent'
                }
            }

            req = urllib.request.Request(
                f"{cls.SUPERVISOR_API}/states/{conf_entity_id}",
                data=json.dumps(conf_data).encode('utf-8'),
                headers=headers,
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                response.read()

            logger.debug(f"Updated HA sensors for {camera_name}")
            return True

        except urllib.error.HTTPError as e:
            logger.warning(f"HA API HTTP error for {camera_name}: {e.code} - {e.reason}")
            return False
        except urllib.error.URLError as e:
            logger.warning(f"HA API connection error: {e.reason}")
            return False
        except Exception as e:
            logger.error(f"HA integration error: {e}")
            return False

    @classmethod
    def remove_sensor(cls, camera_name: str) -> bool:
        """Remove Home Assistant sensor entities for a camera."""
        import urllib.request
        import urllib.error

        token = os.environ.get('SUPERVISOR_TOKEN', '')
        if not token:
            return False

        sanitized_name = cls._sanitize_name(camera_name)
        headers = cls._get_headers()

        entity_ids = [
            f"sensor.camera_ocr_{sanitized_name}_value",
            f"sensor.camera_ocr_{sanitized_name}_text",
            f"sensor.camera_ocr_{sanitized_name}_confidence"
        ]

        for entity_id in entity_ids:
            try:
                # Set state to unavailable to indicate removal
                data = {'state': 'unavailable', 'attributes': {'friendly_name': 'Removed'}}
                req = urllib.request.Request(
                    f"{cls.SUPERVISOR_API}/states/{entity_id}",
                    data=json.dumps(data).encode('utf-8'),
                    headers=headers,
                    method='POST'
                )
                with urllib.request.urlopen(req, timeout=5) as response:
                    response.read()
            except Exception as e:
                logger.debug(f"Could not remove entity {entity_id}: {e}")

        return True


class CameraProcessor:
    """Process camera streams and extract values with template matching support."""

    def __init__(self):
        self.cameras: dict[str, CameraConfig] = {}
        self.values: dict[str, ExtractedValue] = {}
        self.history: dict[str, list] = {}  # Store last 20 values per camera
        self.template_matcher = TemplateMatcher(TEMPLATES_PATH)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.MAX_HISTORY = 20
        self._load_history()  # Load history from persistent storage

    def _load_history(self):
        """Load history from persistent storage."""
        try:
            if os.path.exists(HISTORY_PATH):
                with open(HISTORY_PATH, 'r') as f:
                    self.history = json.load(f)
                logger.info(f"Loaded history for {len(self.history)} cameras from {HISTORY_PATH}")
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            self.history = {}

    def _save_history(self):
        """Save history to persistent storage."""
        try:
            with open(HISTORY_PATH, 'w') as f:
                json.dump(self.history, f)
        except Exception as e:
            logger.error(f"Error saving history: {e}")

    def load_config(self) -> int:
        """Load configuration - first from persistent storage, then from options.json."""
        try:
            cameras_data = []
            scan_interval = 30

            # First try to load from persistent storage (survives reinstalls)
            if os.path.exists(CAMERAS_PATH):
                logger.info(f"Loading cameras from persistent storage: {CAMERAS_PATH}")
                with open(CAMERAS_PATH, 'r') as f:
                    persistent_config = json.load(f)
                cameras_data = persistent_config.get('cameras', [])
                scan_interval = persistent_config.get('scan_interval', 30)

            # If no persistent config, try options.json and migrate
            elif os.path.exists(OPTIONS_PATH):
                logger.info(f"Loading cameras from options.json (first run)")
                with open(OPTIONS_PATH, 'r') as f:
                    options = json.load(f)
                cameras_data = options.get('cameras', [])
                scan_interval = options.get('scan_interval', 30)

                # Migrate to persistent storage
                if cameras_data:
                    self._save_to_persistent({'cameras': cameras_data, 'scan_interval': scan_interval})

            self.cameras.clear()
            for cam_config in cameras_data:
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
                    min_value=cam_config.get('min_value'),
                    max_value=cam_config.get('max_value'),
                )
                self.cameras[camera.name] = camera
                logger.info(f"Loaded camera: {camera.name}")

            return scan_interval

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return 30

    def _save_to_persistent(self, config: dict):
        """Save to persistent storage in /config directory."""
        try:
            with open(CAMERAS_PATH, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Configuration saved to persistent storage: {CAMERAS_PATH}")
        except Exception as e:
            logger.error(f"Error saving to persistent storage: {e}")

    def save_config(self, cameras_list: list = None, settings: dict = None):
        """Save configuration to persistent storage."""
        try:
            # Load existing persistent config
            if os.path.exists(CAMERAS_PATH):
                with open(CAMERAS_PATH, 'r') as f:
                    config = json.load(f)
            else:
                config = {'cameras': [], 'scan_interval': 30, 'log_level': 'info'}

            # Update cameras if provided
            if cameras_list is not None:
                config['cameras'] = cameras_list

            # Update settings if provided
            if settings:
                config.update(settings)

            # Save to persistent storage (survives reinstalls)
            self._save_to_persistent(config)

            # Also update options.json for HA add-on config interface
            try:
                if os.path.exists(OPTIONS_PATH):
                    with open(OPTIONS_PATH, 'r') as f:
                        options = json.load(f)
                    options['cameras'] = config.get('cameras', [])
                    options['scan_interval'] = config.get('scan_interval', 30)
                    with open(OPTIONS_PATH, 'w') as f:
                        json.dump(options, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not update options.json: {e}")

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

    def _open_video_capture(self, url: str) -> cv2.VideoCapture:
        """Open video capture with proper backend and options for RTSP."""
        # Set FFmpeg options for RTSP streams
        if url.lower().startswith('rtsp://'):
            # Use FFmpeg backend with TCP transport (more reliable than UDP)
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|analyzeduration;5000000|probesize;5000000'
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(url)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Set timeout (in milliseconds)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)

        return cap

    def capture_frame(self, camera: CameraConfig) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Capture a frame from camera."""
        try:
            url = self.get_authenticated_url(camera)
            logger.info(f"Capturing frame from: {url[:50]}...")

            cap = self._open_video_capture(url)

            if not cap.isOpened():
                logger.error(f"Failed to open stream: {url[:50]}...")
                return None, "Failed to open stream. Check URL, credentials and network connectivity."

            # Read a few frames to get latest
            frame = None
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    break

            cap.release()

            if not ret or frame is None:
                logger.error(f"Failed to read frame from: {url[:50]}...")
                return None, "Stream opened but failed to read frame"

            logger.info(f"Successfully captured frame: {frame.shape}")
            return frame, None

        except Exception as e:
            logger.exception(f"Error capturing frame: {e}")
            return None, str(e)

    def capture_frame_from_url(self, url: str, username: str = "", password: str = "") -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Capture a frame from any URL."""
        try:
            full_url = url
            if username and password and "@" not in url and "://" in url:
                protocol, rest = url.split("://", 1)
                full_url = f"{protocol}://{username}:{password}@{rest}"

            logger.info(f"Capturing frame from URL: {full_url[:50]}...")

            cap = self._open_video_capture(full_url)

            if not cap.isOpened():
                logger.error(f"Failed to open stream: {full_url[:50]}...")
                return None, "Failed to open stream. Check URL, credentials and network connectivity."

            frame = None
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    break

            cap.release()

            if not ret or frame is None:
                logger.error(f"Failed to read frame from: {full_url[:50]}...")
                return None, "Stream opened but failed to read frame"

            logger.info(f"Successfully captured frame: {frame.shape}")
            return frame, None

        except Exception as e:
            logger.exception(f"Error capturing frame: {e}")
            return None, str(e)

    def preprocess_image(self, image: np.ndarray, method: str) -> np.ndarray:
        """Preprocess image for OCR."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Add padding to prevent digit cutoff at edges
        pad = 10
        gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

        # Scale up small images significantly for better OCR
        h, w = gray.shape[:2]
        if h < 60 or w < 60:
            scale = max(250 / h, 250 / w, 5)
        elif h < 120 or w < 120:
            scale = max(200 / h, 200 / w, 4)
        else:
            scale = 3
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Enhance contrast for digital displays
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        if method == "none":
            return gray
        elif method == "threshold":
            _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Morphological cleanup for digits
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
            return processed
        elif method == "adaptive":
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 15, 3)
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
            return processed
        elif method == "invert":
            _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed = cv2.bitwise_not(processed)
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
            return processed
        else:  # auto - try multiple methods and return best
            # Apply CLAHE for better contrast on digital displays
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Denoise while preserving edges
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

            # Sharpen to enhance digit edges
            kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)

            # Try Otsu's threshold
            blurred = cv2.GaussianBlur(sharpened, (3, 3), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Morphological cleanup to connect digit segments
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Determine if we need to invert based on the mean
            if np.mean(sharpened) < 127:
                return cv2.bitwise_not(binary)
            return binary

    def try_multiple_preprocessing(self, image: np.ndarray) -> list:
        """Try multiple preprocessing methods and return all results."""
        methods = ['auto', 'threshold', 'adaptive', 'invert']
        results = []
        for method in methods:
            try:
                processed = self.preprocess_image(image, method)
                results.append((method, processed))
            except:
                pass
        return results

    def _ocr_single(self, processed: np.ndarray, psm: int = 7) -> tuple:
        """Run OCR on a single preprocessed image."""
        config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789.-"
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

        return value, raw_text, avg_confidence

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

            # Try multiple preprocessing methods and PSM modes to find best result
            best_result = (None, "", 0)  # (value, raw_text, confidence)
            methods_to_try = ['auto', 'threshold', 'adaptive', 'invert'] if camera.preprocessing == 'auto' else [camera.preprocessing]
            psm_modes = [7, 8, 6, 13]  # 7=single line, 8=single word, 6=uniform block, 13=raw line

            for method in methods_to_try:
                processed = self.preprocess_image(roi_frame, method)

                for psm in psm_modes:
                    try:
                        value, raw_text, confidence = self._ocr_single(processed, psm)

                        # Keep best result based on confidence and having a valid value
                        if value is not None and confidence > best_result[2]:
                            best_result = (value, raw_text, confidence)
                            logger.debug(f"Better result: {value} ({confidence:.0f}%) with method={method}, psm={psm}")

                        # If we got high confidence, stop trying
                        if confidence > 80 and value is not None:
                            break
                    except Exception as e:
                        logger.debug(f"OCR attempt failed: {e}")
                        continue

                if best_result[2] > 80:
                    break

            value, raw_text, avg_confidence = best_result

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
            # Update Home Assistant sensors with error state
            HomeAssistantIntegration.update_sensor(
                camera_name=camera.name,
                value=None,
                raw_text="",
                confidence=0,
                unit=camera.unit,
                error=error
            )
            return

        result = self.extract_value(camera, frame)

        # Filter out values outside expected range (if configured)
        if result.value is not None:
            out_of_range = False
            if camera.min_value is not None and result.value < camera.min_value:
                logger.info(f"{camera.name}: Value {result.value} below min {camera.min_value}, ignoring")
                out_of_range = True
            if camera.max_value is not None and result.value > camera.max_value:
                logger.info(f"{camera.name}: Value {result.value} above max {camera.max_value}, ignoring")
                out_of_range = True
            if out_of_range:
                result.error = f"Value {result.value} out of expected range ({camera.min_value or '-∞'} to {camera.max_value or '+∞'})"
                result.value = None

        # Generate AI scene description if enabled
        ai_config = AIService.load_config()
        if ai_config.enabled_for_description:
            try:
                _, buffer = cv2.imencode('.jpg', frame)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                result.video_description = AIService.describe_scene(image_base64)
            except Exception as e:
                logger.error(f"AI scene description error: {e}")

        self.values[camera.name] = result

        # Update Home Assistant sensors
        HomeAssistantIntegration.update_sensor(
            camera_name=camera.name,
            value=result.value,
            raw_text=result.raw_text,
            confidence=result.confidence,
            unit=camera.unit,
            error=result.error or "",
            video_description=result.video_description
        )

        # Add to history
        if camera.name not in self.history:
            self.history[camera.name] = []

        # Save ROI image for history
        history_id = f"{int(result.timestamp * 1000)}"
        roi_image_id = None
        try:
            # Extract ROI area from frame
            if camera.roi_width > 0 and camera.roi_height > 0:
                x, y = camera.roi_x, camera.roi_y
                w, h = camera.roi_width, camera.roi_height
                img_h, img_w = frame.shape[:2]
                x = max(0, min(x, img_w - 1))
                y = max(0, min(y, img_h - 1))
                w = min(w, img_w - x)
                h = min(h, img_h - y)
                roi_frame = frame[y:y+h, x:x+w]
            else:
                roi_frame = frame

            # Save ROI thumbnail (resize to save space)
            max_dim = 200
            roi_h, roi_w = roi_frame.shape[:2]
            if roi_w > max_dim or roi_h > max_dim:
                scale = max_dim / max(roi_w, roi_h)
                new_w, new_h = int(roi_w * scale), int(roi_h * scale)
                roi_frame = cv2.resize(roi_frame, (new_w, new_h))

            # Save to history images folder
            safe_name = re.sub(r'[^\w\-]', '_', camera.name)
            camera_img_path = Path(HISTORY_IMAGES_PATH) / safe_name
            camera_img_path.mkdir(parents=True, exist_ok=True)
            img_path = camera_img_path / f"{history_id}.jpg"
            cv2.imwrite(str(img_path), roi_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            roi_image_id = history_id
        except Exception as e:
            logger.debug(f"Failed to save history ROI image: {e}")

        # Get AI provider name if used
        ai_provider = None
        ai_config = AIService.load_config()
        if ai_config.enabled_for_ocr:
            ai_provider = ai_config.provider

        self.history[camera.name].append({
            'value': result.value,
            'timestamp': result.timestamp,
            'confidence': result.confidence,
            'raw_text': result.raw_text,
            'error': result.error,
            'video_description': result.video_description,
            'roi_image_id': roi_image_id,
            'ocr_provider': ai_provider or 'tesseract'
        })

        # Keep only last MAX_HISTORY entries and clean up old images
        if len(self.history[camera.name]) > self.MAX_HISTORY:
            old_entries = self.history[camera.name][:-self.MAX_HISTORY]
            self.history[camera.name] = self.history[camera.name][-self.MAX_HISTORY:]
            # Clean up old images
            safe_name = re.sub(r'[^\w\-]', '_', camera.name)
            for entry in old_entries:
                if entry.get('roi_image_id'):
                    try:
                        old_img = Path(HISTORY_IMAGES_PATH) / safe_name / f"{entry['roi_image_id']}.jpg"
                        if old_img.exists():
                            old_img.unlink()
                    except:
                        pass

        # Save history to persistent storage
        self._save_history()

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
    <title>Camera OCR</title>
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
        .value-card-status.pending { background: rgba(255, 193, 7, 0.2); color: #ffc107; }

        .value-card-value {
            font-size: 42px;
            font-weight: 700;
            line-height: 1;
            margin-bottom: 8px;
        }

        .value-card-unit { font-size: 18px; color: var(--text-2); margin-left: 4px; }
        .value-card-meta { font-size: 12px; color: var(--text-3); }
        .value-card-description {
            font-size: 12px;
            color: var(--text-2);
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid var(--border);
            font-style: italic;
        }

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

        /* URL Mode Toggle */
        .url-mode-toggle {
            display: flex;
            gap: 0;
            margin-top: 8px;
        }
        .url-mode-btn {
            flex: 1;
            padding: 8px 16px;
            border: 1px solid var(--border);
            background: var(--card-bg);
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }
        .url-mode-btn:first-child {
            border-radius: 6px 0 0 6px;
        }
        .url-mode-btn:last-child {
            border-radius: 0 6px 6px 0;
            border-left: none;
        }
        .url-mode-btn:hover {
            background: var(--bg);
        }
        .url-mode-btn.active {
            background: var(--primary);
            border-color: var(--primary);
            color: white;
        }
        .form-hint {
            display: block;
            margin-top: 4px;
            font-size: 11px;
            color: var(--text-secondary);
        }
        .url-preview {
            background: var(--bg);
            border-radius: 6px;
            padding: 12px;
            border: 1px solid var(--border);
        }
        .url-preview-text {
            font-family: monospace;
            font-size: 12px;
            color: var(--primary);
            word-break: break-all;
        }

        /* Preview Container */
        .preview-container {
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 20px;
            min-height: 500px;
        }

        .preview-main {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        @media (max-width: 1000px) {
            .preview-container { grid-template-columns: 1fr; }
        }

        .preview-frame {
            background: var(--bg-3);
            border-radius: 8px;
            overflow: auto;
            position: relative;
            max-height: 500px;
            flex-shrink: 0;
        }

        .preview-wrapper {
            position: relative;
            display: inline-block;
            transform-origin: top left;
        }

        .preview-frame img {
            display: block;
            max-width: none;
        }

        .preview-canvas {
            position: absolute;
            top: 0;
            left: 0;
            cursor: crosshair;
        }

        .saved-rois-section {
            margin-top: 12px;
            padding: 12px;
            background: var(--card-bg);
            border-radius: 8px;
        }

        .saved-rois-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            font-weight: 500;
            color: var(--text-2);
        }

        .saved-rois-list {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }

        .saved-roi-item {
            position: relative;
            width: 120px;
            background: var(--bg);
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid var(--border);
        }

        .saved-roi-item img {
            width: 100%;
            height: 80px;
            object-fit: cover;
            display: block;
        }

        .saved-roi-info {
            padding: 6px;
            font-size: 11px;
        }

        .saved-roi-value {
            font-weight: 600;
            color: var(--primary);
            font-size: 14px;
        }

        .saved-roi-time {
            color: var(--text-3);
            margin-top: 2px;
        }

        .saved-roi-item.validated {
            border-color: var(--success);
        }

        .validated-badge {
            position: absolute;
            top: 4px;
            left: 4px;
            background: var(--success);
            color: #fff;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            font-weight: bold;
            z-index: 2;
        }

        .saved-roi-delete {
            position: absolute;
            top: 4px;
            right: 4px;
            width: 20px;
            height: 20px;
            border: none;
            background: rgba(244, 67, 54, 0.9);
            color: white;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            line-height: 1;
            opacity: 0;
            transition: opacity 0.2s;
        }

        .saved-roi-item:hover .saved-roi-delete {
            opacity: 1;
        }

        .saved-roi-delete:hover {
            background: var(--error);
        }

        .saved-roi-buttons {
            position: absolute;
            top: 4px;
            left: 4px;
            display: flex;
            gap: 4px;
            opacity: 0;
            transition: opacity 0.2s;
        }

        .saved-roi-item:hover .saved-roi-buttons {
            opacity: 1;
        }

        .saved-roi-apply, .saved-roi-test {
            width: 24px;
            height: 24px;
            padding: 0;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .saved-roi-apply {
            background: rgba(76, 175, 80, 0.9);
            color: white;
        }

        .saved-roi-test {
            background: rgba(33, 150, 243, 0.9);
            color: white;
        }

        .saved-roi-apply:hover {
            background: var(--success);
        }

        .saved-roi-test:hover {
            background: var(--primary);
        }

        .zoom-controls {
            position: absolute;
            top: 10px;
            right: 10px;
            display: flex;
            gap: 4px;
            z-index: 10;
            background: rgba(0,0,0,0.7);
            border-radius: 6px;
            padding: 4px;
        }

        .zoom-btn {
            width: 32px;
            height: 32px;
            border: none;
            background: var(--card-bg);
            color: var(--text);
            border-radius: 4px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            font-weight: bold;
        }

        .zoom-btn:hover {
            background: var(--primary);
            color: white;
        }

        .ptz-controls {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 2px;
            margin-left: 8px;
            padding-left: 8px;
            border-left: 1px solid rgba(255,255,255,0.2);
        }

        .ptz-row {
            display: flex;
            gap: 2px;
        }

        .ptz-btn {
            width: 28px;
            height: 28px;
            font-size: 14px;
        }

        .ptz-home {
            font-size: 16px;
        }

        .zoom-level {
            color: white;
            font-size: 12px;
            padding: 0 8px;
            display: flex;
            align-items: center;
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
            flex-direction: column;
            gap: 12px;
        }
        .discovery-item-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: 100%;
        }
        .discovery-item-info h4 { font-size: 15px; margin-bottom: 4px; }
        .discovery-item-info p { font-size: 13px; color: var(--text-3); }
        .discovery-item-preview {
            width: 100%;
            aspect-ratio: 16/9;
            background: var(--bg);
            border-radius: 6px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }
        .discovery-item-preview img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .discovery-item-preview .preview-placeholder {
            color: var(--text-3);
            font-size: 12px;
            text-align: center;
            height: auto;
            padding: 20px;
        }
        .discovery-item-preview .preview-placeholder svg {
            width: 48px;
            height: 48px;
            margin-bottom: 8px;
            opacity: 0.4;
        }
        .discovery-item-preview .preview-error {
            color: var(--text-3);
            font-size: 11px;
            text-align: center;
            padding: 16px;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
        }
        .discovery-item-preview .preview-error svg {
            width: 40px;
            height: 40px;
            opacity: 0.3;
        }
        .discovery-item-preview .preview-error small {
            opacity: 0.7;
        }
        .discovery-item-actions {
            display: flex;
            gap: 8px;
            width: 100%;
        }
        .discovery-item-actions .btn {
            flex: 1;
        }

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

        .live-badge.paused {
            background: rgba(128, 128, 128, 0.2);
            color: var(--text-secondary);
        }
        .live-badge.paused::before {
            background: var(--text-secondary);
            animation: none;
        }
        .live-controls {
            display: inline-flex;
            margin-left: 8px;
            gap: 4px;
        }
        .live-btn {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 4px 8px;
            cursor: pointer;
            color: var(--text-secondary);
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }
        .live-btn:hover {
            background: var(--bg-2);
            color: var(--text);
        }

        /* History table */
        .history-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        .history-table th, .history-table td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        .history-table th {
            background: var(--bg);
            font-weight: 600;
            color: var(--text-secondary);
            font-size: 12px;
            text-transform: uppercase;
        }
        .history-table tr:hover {
            background: var(--bg);
        }
        .history-table td.value-cell {
            font-weight: 600;
            color: var(--primary);
            text-align: right;
        }
        .history-table td.value-cell span {
            font-weight: 600;
        }
        .history-table th.value-header {
            text-align: right;
        }
        .history-table .order-cell,
        .history-table th.order-header {
            width: 40px;
            text-align: center;
            color: var(--text-secondary);
        }
        .history-table .error-cell {
            color: var(--error);
        }
        .low-confidence {
            color: var(--error) !important;
        }
        .low-confidence-text {
            font-size: 11px;
            color: var(--error);
            opacity: 0.8;
        }
        .confidence-bar {
            width: 80px;
            height: 8px;
            background: var(--bg);
            border-radius: 4px;
            overflow: hidden;
            display: inline-block;
            vertical-align: middle;
        }
        .confidence-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        .confidence-bar-fill.high { background: var(--success); }
        .confidence-bar-fill.medium { background: var(--warning, #f59e0b); }
        .confidence-bar-fill.low { background: var(--error); }
        .confidence-text {
            display: inline-block;
            width: 35px;
            text-align: right;
            margin-right: 8px;
            font-size: 12px;
        }
        .history-thumbnail {
            width: 50px;
            height: 35px;
            object-fit: cover;
            border-radius: 4px;
            cursor: pointer;
            border: 1px solid var(--border);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .history-thumbnail:hover {
            transform: scale(1.1);
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        .history-thumbnail-placeholder {
            width: 50px;
            height: 35px;
            background: var(--bg);
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-3);
            font-size: 10px;
        }
        .history-detail-dialog {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1001;
            padding: 20px;
        }
        .history-detail-content {
            background: var(--card-bg);
            border-radius: 12px;
            max-width: 600px;
            width: 100%;
            max-height: 90vh;
            overflow-y: auto;
        }
        .history-detail-header {
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .history-detail-header h3 {
            margin: 0;
            font-size: 16px;
        }
        .history-detail-body {
            padding: 20px;
        }
        .history-detail-image {
            width: 100%;
            max-height: 300px;
            object-fit: contain;
            border-radius: 8px;
            background: var(--bg);
            margin-bottom: 16px;
        }
        .history-detail-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        .history-detail-item {
            background: var(--bg);
            padding: 12px;
            border-radius: 6px;
        }
        .history-detail-item label {
            font-size: 11px;
            color: var(--text-3);
            text-transform: uppercase;
            display: block;
            margin-bottom: 4px;
        }
        .history-detail-item .value {
            font-size: 15px;
            font-weight: 600;
        }
        .history-detail-item.full-width {
            grid-column: 1 / -1;
        }
        .history-tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }
        .history-tab {
            padding: 6px 14px;
            border: 1px solid var(--border);
            background: var(--card-bg);
            color: var(--text-secondary);
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
        }
        .history-tab:hover {
            background: var(--bg);
        }
        .history-tab.active {
            background: var(--primary);
            border-color: var(--primary);
            color: white;
        }

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
                Camera OCR
            </div>
            <nav class="nav">
                <button class="nav-btn active" data-page="dashboard">Dashboard</button>
                <button class="nav-btn" data-page="cameras">Cameras</button>
                <button class="nav-btn" data-page="live">Live Preview</button>
                <button class="nav-btn" data-page="templates">Templates</button>
                <button class="nav-btn" data-page="discover">Discover</button>
                <button class="nav-btn" data-page="ai-settings">AI Settings</button>
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
                        <span class="live-badge" id="live-badge">LIVE</span>
                        <div class="live-controls">
                            <button class="live-btn" id="pause-btn" onclick="toggleLiveUpdates()" title="Pause updates">
                                <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
                                    <rect x="6" y="4" width="4" height="16"></rect>
                                    <rect x="14" y="4" width="4" height="16"></rect>
                                </svg>
                            </button>
                            <button class="live-btn" id="resume-btn" onclick="toggleLiveUpdates()" title="Resume updates" style="display: none;">
                                <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
                                    <polygon points="5,3 19,12 5,21"></polygon>
                                </svg>
                            </button>
                        </div>
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

                <!-- Value History Section -->
                <div class="card" style="margin-top: 20px;">
                    <div class="card-title">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
                            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                        </svg>
                        Value History (Last 20 readings)
                    </div>
                    <div id="history-container">
                        <p style="color: var(--text-3);">No history available yet</p>
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
                        <div class="form-group" style="display: flex; align-items: flex-end; gap: 8px;">
                            <button class="btn btn-primary" onclick="refreshLivePreview()">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
                                    <polyline points="23 4 23 10 17 10"></polyline>
                                    <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
                                </svg>
                                Refresh
                            </button>
                            <button class="btn btn-secondary" onclick="editSelectedCamera()" title="Edit camera settings">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
                                    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                                    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                                </svg>
                                Edit
                            </button>
                        </div>
                    </div>

                    <div class="preview-container">
                        <div class="preview-main">
                            <div class="preview-frame" id="preview-frame">
                                <div class="preview-placeholder" id="preview-placeholder">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                                        <circle cx="12" cy="13" r="4"></circle>
                                    </svg>
                                    <p>Select a camera and click Refresh to load preview</p>
                                </div>
                                <div class="zoom-controls" id="zoom-controls" style="display: none;">
                                    <button class="zoom-btn" onclick="zoomOut()" title="Zoom Out">−</button>
                                    <span class="zoom-level" id="zoom-level">100%</span>
                                    <button class="zoom-btn" onclick="zoomIn()" title="Zoom In">+</button>
                                    <button class="zoom-btn" onclick="resetZoom()" title="Reset Zoom">⟲</button>
                                    <div class="ptz-controls">
                                        <button class="zoom-btn ptz-btn" onclick="ptzMove('up')" title="Tilt Up">▲</button>
                                        <div class="ptz-row">
                                            <button class="zoom-btn ptz-btn" onclick="ptzMove('left')" title="Pan Left">◄</button>
                                            <button class="zoom-btn ptz-btn ptz-home" onclick="ptzMove('home')" title="Home">⌂</button>
                                            <button class="zoom-btn ptz-btn" onclick="ptzMove('right')" title="Pan Right">►</button>
                                        </div>
                                        <button class="zoom-btn ptz-btn" onclick="ptzMove('down')" title="Tilt Down">▼</button>
                                    </div>
                                </div>
                                <div class="preview-wrapper" id="preview-wrapper">
                                    <img id="preview-image" style="display: none;" />
                                    <canvas id="preview-canvas" class="preview-canvas" style="display: none;"></canvas>
                                </div>
                            </div>

                            <!-- Saved ROIs Section - Outside scrollable area -->
                            <div class="saved-rois-section" id="saved-rois-section" style="display: none;">
                                <div class="saved-rois-header">
                                    <span>Saved ROIs</span>
                                    <div style="display: flex; gap: 8px;">
                                        <button class="btn btn-sm" onclick="trainOCR()" id="train-ocr-btn" style="background: var(--warning); color: #000;" title="Train OCR using validated ROIs">
                                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14" style="margin-right: 4px;">
                                                <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path>
                                                <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path>
                                            </svg>
                                            Train OCR
                                        </button>
                                        <button class="btn btn-primary btn-sm" onclick="testAllROIs()" id="test-all-btn">Test All</button>
                                        <button class="btn btn-success btn-sm" onclick="saveCurrentROI()">Save Current</button>
                                    </div>
                                </div>
                                <div class="saved-rois-list" id="saved-rois-list"></div>
                            </div>
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
                                    <div id="result-roi-preview" style="display: none; margin-bottom: 12px; border-radius: 6px; overflow: hidden; background: var(--bg);"></div>
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

            <!-- AI Settings Page -->
            <div id="page-ai-settings" class="page">
                <div class="card">
                    <div class="card-title">AI Provider Configuration</div>
                    <p style="font-size: 13px; color: var(--text-3); margin-bottom: 16px;">
                        Configure an AI provider to enhance OCR accuracy and generate video scene descriptions.
                    </p>

                    <div class="form-group">
                        <label class="form-label">AI Provider</label>
                        <select id="ai-provider" class="form-input" onchange="onAIProviderChange()">
                            <option value="none">None (Disabled)</option>
                            <optgroup label="AI Vision Models">
                                <option value="openai">OpenAI (GPT-4o)</option>
                                <option value="anthropic">Anthropic (Claude)</option>
                                <option value="google">Google (Gemini)</option>
                                <option value="ollama">Ollama (Local)</option>
                                <option value="custom">Custom (OpenAI-compatible)</option>
                            </optgroup>
                            <optgroup label="Cloud OCR Services">
                                <option value="google-vision">Google Cloud Vision</option>
                                <option value="google-docai">Google Document AI</option>
                                <option value="azure-ocr">Azure Computer Vision</option>
                                <option value="aws-textract">AWS Textract</option>
                            </optgroup>
                        </select>
                    </div>

                    <div id="ai-config-fields" style="display: none;">
                        <div class="form-group" id="ai-apikey-group">
                            <label class="form-label">API Key</label>
                            <input type="password" id="ai-apikey" class="form-input" placeholder="Enter API key">
                            <p style="font-size: 12px; color: var(--text-3); margin-top: 4px;">
                                Your API key is stored locally and never shared.
                            </p>
                        </div>

                        <div class="form-group" id="ai-url-group" style="display: none;">
                            <label class="form-label" id="ai-url-label">API URL</label>
                            <input type="text" id="ai-url" class="form-input" placeholder="http://localhost:11434">
                            <p style="font-size: 12px; color: var(--text-3); margin-top: 4px;" id="ai-url-hint">
                                URL to your Ollama or compatible server.
                            </p>
                        </div>

                        <div class="form-group" id="ai-region-group" style="display: none;">
                            <label class="form-label">AWS Region</label>
                            <input type="text" id="ai-region" class="form-input" placeholder="us-east-1">
                            <p style="font-size: 12px; color: var(--text-3); margin-top: 4px;">
                                AWS region for Textract (e.g., us-east-1, eu-west-1)
                            </p>
                        </div>

                        <div class="form-group">
                            <label class="form-label">Model (optional)</label>
                            <input type="text" id="ai-model" class="form-input" placeholder="Leave empty for default">
                            <p style="font-size: 12px; color: var(--text-3); margin-top: 4px;" id="ai-model-hint">
                                Default: gpt-4o
                            </p>
                        </div>

                        <div class="form-group">
                            <label class="form-label">Features</label>
                            <div style="display: flex; flex-direction: column; gap: 8px;">
                                <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                                    <input type="checkbox" id="ai-enable-ocr">
                                    <span>Use AI to enhance OCR accuracy</span>
                                </label>
                                <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                                    <input type="checkbox" id="ai-enable-description">
                                    <span>Generate video scene descriptions</span>
                                </label>
                            </div>
                        </div>
                    </div>

                    <div style="display: flex; gap: 8px; margin-top: 16px;">
                        <button class="btn btn-primary" onclick="saveAIConfig()">Save Configuration</button>
                    </div>
                </div>

                <!-- Test AI Provider Section -->
                <div class="card" id="ai-test-section" style="display: none;">
                    <div class="card-title">Test AI Provider</div>
                    <p style="font-size: 13px; color: var(--text-3); margin-bottom: 16px;">
                        Test the configured AI/OCR provider with an image.
                    </p>

                    <div class="form-group">
                        <label class="form-label">Image Source</label>
                        <select id="ai-test-source" class="form-input" onchange="onAITestSourceChange()">
                            <option value="upload">Upload Image</option>
                            <option value="camera">Camera Live Preview</option>
                        </select>
                    </div>

                    <div id="ai-test-upload-section">
                        <div class="form-group">
                            <label class="form-label">Select Image</label>
                            <input type="file" id="ai-test-file" accept="image/*" class="form-input" onchange="onAITestFileSelect()">
                        </div>
                        <div id="ai-test-preview-upload" style="margin-bottom: 16px; display: none;">
                            <img id="ai-test-preview-img" style="max-width: 100%; max-height: 200px; border-radius: 6px; border: 1px solid var(--border);">
                        </div>
                    </div>

                    <div id="ai-test-camera-section" style="display: none;">
                        <div class="form-group">
                            <label class="form-label">Select Camera</label>
                            <select id="ai-test-camera" class="form-input" onchange="onAITestCameraChange()">
                                <option value="">-- Select Camera --</option>
                            </select>
                        </div>
                        <div id="ai-test-preview-camera" style="margin-bottom: 16px; display: none;">
                            <p style="font-size: 12px; color: var(--text-3); margin-bottom: 8px;">Camera Preview (ROI area highlighted):</p>
                            <img id="ai-test-camera-img" style="max-width: 100%; max-height: 200px; border-radius: 6px; border: 1px solid var(--border);">
                        </div>
                    </div>

                    <button class="btn btn-primary" onclick="testAIProvider()" id="test-ai-btn">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
                            <polygon points="5 3 19 12 5 21 5 3"></polygon>
                        </svg>
                        Test AI Provider
                    </button>
                </div>

                <div class="card" id="ai-test-results" style="display: none;">
                    <div class="card-title">Test Results</div>
                    <div id="ai-test-content"></div>
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

                <!-- URL Input Mode Toggle -->
                <div class="form-group">
                    <label class="form-label">Stream URL *</label>
                    <div class="url-mode-toggle">
                        <button type="button" class="url-mode-btn active" id="url-mode-full" onclick="setUrlMode('full')">Full URL</button>
                        <button type="button" class="url-mode-btn" id="url-mode-build" onclick="setUrlMode('build')">Build URL</button>
                    </div>
                </div>

                <!-- Full URL Input -->
                <div class="form-group url-input-full" id="url-input-full">
                    <input type="text" id="camera-url" class="form-input" placeholder="rtsp://192.168.1.100:554/stream1" oninput="parseUrlToFields()">
                    <small class="form-hint">Enter full RTSP/HTTP URL (credentials will be extracted automatically)</small>
                </div>

                <!-- Build URL Inputs -->
                <div class="url-input-build" id="url-input-build" style="display: none;">
                    <div class="form-row">
                        <div class="form-group" style="flex: 0 0 100px;">
                            <label class="form-label">Protocol</label>
                            <select id="camera-protocol" class="form-select" onchange="buildUrlFromFields()">
                                <option value="rtsp">RTSP</option>
                                <option value="http">HTTP</option>
                                <option value="https">HTTPS</option>
                            </select>
                        </div>
                        <div class="form-group" style="flex: 1;">
                            <label class="form-label">Host/IP *</label>
                            <input type="text" id="camera-host" class="form-input" placeholder="192.168.1.100" oninput="buildUrlFromFields()">
                        </div>
                        <div class="form-group" style="flex: 0 0 80px;">
                            <label class="form-label">Port</label>
                            <input type="text" id="camera-port" class="form-input" placeholder="554" oninput="buildUrlFromFields()">
                        </div>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Stream Path</label>
                        <input type="text" id="camera-path" class="form-input" placeholder="/stream1" oninput="buildUrlFromFields()">
                        <small class="form-hint">Common paths: /stream1, /Streaming/Channels/101, /cam/realmonitor?channel=1&subtype=0</small>
                    </div>
                </div>

                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">Username</label>
                        <input type="text" id="camera-username" class="form-input" oninput="buildUrlFromFields()">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Password</label>
                        <input type="password" id="camera-password" class="form-input" oninput="buildUrlFromFields()">
                    </div>
                </div>

                <!-- Generated URL Preview -->
                <div class="form-group url-preview" id="url-preview" style="display: none;">
                    <label class="form-label">Generated URL</label>
                    <div class="url-preview-text" id="url-preview-text"></div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">Value Name</label>
                        <input type="text" id="camera-value-name" class="form-input" placeholder="Temperature" value="Value">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Unit</label>
                        <select id="camera-unit" class="form-select">
                            <option value="">None</option>
                            <option value="°C">°C (Celsius)</option>
                            <option value="°F">°F (Fahrenheit)</option>
                            <option value="K">K (Kelvin)</option>
                            <option value="bar">bar (Pressure)</option>
                            <option value="psi">psi (Pressure)</option>
                            <option value="kPa">kPa (Pressure)</option>
                            <option value="%">% (Percentage)</option>
                            <option value="V">V (Voltage)</option>
                            <option value="A">A (Amperage)</option>
                            <option value="W">W (Watts)</option>
                            <option value="kWh">kWh (Energy)</option>
                            <option value="L">L (Liters)</option>
                            <option value="m³">m³ (Cubic meters)</option>
                            <option value="Hz">Hz (Frequency)</option>
                            <option value="RPM">RPM (Rotation)</option>
                        </select>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">Min Value (Optional)</label>
                        <input type="number" id="camera-min-value" class="form-input" placeholder="e.g., 0" step="any">
                        <p style="font-size: 11px; color: var(--text-3); margin-top: 4px;">Ignore readings below this value</p>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Max Value (Optional)</label>
                        <input type="number" id="camera-max-value" class="form-input" placeholder="e.g., 100" step="any">
                        <p style="font-size: 11px; color: var(--text-3); margin-top: 4px;">Ignore readings above this value</p>
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
        let zoomLevel = 1;
        let liveUpdateInterval = null;
        let isLivePaused = false;

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            loadData();
            startLiveUpdates();
            setupCanvas();
            setupZoomWheel();
        });

        function startLiveUpdates() {
            if (liveUpdateInterval) clearInterval(liveUpdateInterval);
            liveUpdateInterval = setInterval(loadValues, 5000);
        }

        function stopLiveUpdates() {
            if (liveUpdateInterval) {
                clearInterval(liveUpdateInterval);
                liveUpdateInterval = null;
            }
        }

        function toggleLiveUpdates() {
            isLivePaused = !isLivePaused;
            const badge = document.getElementById('live-badge');
            const pauseBtn = document.getElementById('pause-btn');
            const resumeBtn = document.getElementById('resume-btn');

            if (isLivePaused) {
                stopLiveUpdates();
                badge.textContent = 'PAUSED';
                badge.classList.add('paused');
                pauseBtn.style.display = 'none';
                resumeBtn.style.display = 'inline-flex';
            } else {
                startLiveUpdates();
                loadValues(); // Refresh immediately
                badge.textContent = 'LIVE';
                badge.classList.remove('paused');
                pauseBtn.style.display = 'inline-flex';
                resumeBtn.style.display = 'none';
            }
        }

        // Zoom & PTZ Functions
        function setupZoomWheel() {
            const frame = document.getElementById('preview-frame');
            frame.addEventListener('wheel', (e) => {
                if (!previewImage) return;
                e.preventDefault();
                if (e.deltaY < 0) {
                    zoomIn();
                } else {
                    zoomOut();
                }
            }, { passive: false });
        }

        function zoomIn() {
            if (zoomLevel < 4) {
                zoomLevel = Math.min(4, zoomLevel + 0.25);
                applyZoom();
            }
        }

        function zoomOut() {
            if (zoomLevel > 0.25) {
                zoomLevel = Math.max(0.25, zoomLevel - 0.25);
                applyZoom();
            }
        }

        function resetZoom() {
            zoomLevel = 1;
            applyZoom();
        }

        async function ptzMove(direction) {
            const select = document.getElementById('preview-camera-select');
            const cameraName = select.value;
            if (!cameraName) {
                toast('Please select a camera first', 'error');
                return;
            }

            try {
                const res = await fetch('api/ptz', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ camera: cameraName, direction: direction })
                });
                const result = await res.json();
                if (result.error) {
                    toast(result.error, 'error');
                } else {
                    toast(`PTZ: ${direction}`, 'success');
                    // Refresh preview after PTZ move
                    setTimeout(() => refreshPreview(), 500);
                }
            } catch (e) {
                toast('PTZ command failed: ' + e.message, 'error');
            }
        }

        function applyZoom() {
            const wrapper = document.getElementById('preview-wrapper');
            const img = document.getElementById('preview-image');
            const canvas = document.getElementById('preview-canvas');

            if (!previewImage) return;

            // Apply zoom
            const scaledWidth = previewImage.width * zoomLevel;
            const scaledHeight = previewImage.height * zoomLevel;

            img.style.width = scaledWidth + 'px';
            img.style.height = scaledHeight + 'px';

            wrapper.style.width = scaledWidth + 'px';
            wrapper.style.height = scaledHeight + 'px';

            // Update canvas
            canvas.width = scaledWidth;
            canvas.height = scaledHeight;
            canvas.style.width = scaledWidth + 'px';
            canvas.style.height = scaledHeight + 'px';

            // Update scale factor for ROI calculations
            imageScale = zoomLevel;

            // Update zoom level display
            document.getElementById('zoom-level').textContent = Math.round(zoomLevel * 100) + '%';

            drawROI();
        }

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
            if (page === 'ai-settings') loadAIConfig();
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
                loadHistory();
            } catch (e) {
                console.error('Failed to load values:', e);
            }
        }

        let historyData = {};
        let selectedHistoryCamera = null;
        let historyViewMode = 'table';  // 'table', 'card', 'chart'

        async function loadHistory() {
            try {
                const res = await fetch('api/history');
                historyData = await res.json();
                renderHistory();
            } catch (e) {
                console.error('Failed to load history:', e);
            }
        }

        function renderHistory() {
            const container = document.getElementById('history-container');
            const cameraNames = Object.keys(historyData);

            if (cameraNames.length === 0) {
                container.innerHTML = '<p style="color: var(--text-3);">No history available yet</p>';
                return;
            }

            // Auto-select first camera if none selected
            if (!selectedHistoryCamera || !historyData[selectedHistoryCamera]) {
                selectedHistoryCamera = cameraNames[0];
            }

            // Render tabs for each camera
            const tabs = cameraNames.map(name => `
                <button class="history-tab ${name === selectedHistoryCamera ? 'active' : ''}"
                        onclick="selectHistoryCamera('${name}')">${name}</button>
            `).join('');

            // View mode toggle
            const viewModeHtml = `
                <div style="display: flex; gap: 4px; margin-left: auto;">
                    <button class="btn btn-sm ${historyViewMode === 'table' ? 'btn-primary' : 'btn-secondary'}" onclick="setHistoryViewMode('table')" title="Table View">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
                            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                            <line x1="3" y1="9" x2="21" y2="9"></line>
                            <line x1="3" y1="15" x2="21" y2="15"></line>
                            <line x1="9" y1="3" x2="9" y2="21"></line>
                        </svg>
                    </button>
                    <button class="btn btn-sm ${historyViewMode === 'card' ? 'btn-primary' : 'btn-secondary'}" onclick="setHistoryViewMode('card')" title="Card View">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
                            <rect x="3" y="3" width="7" height="7"></rect>
                            <rect x="14" y="3" width="7" height="7"></rect>
                            <rect x="14" y="14" width="7" height="7"></rect>
                            <rect x="3" y="14" width="7" height="7"></rect>
                        </svg>
                    </button>
                    <button class="btn btn-sm ${historyViewMode === 'chart' ? 'btn-primary' : 'btn-secondary'}" onclick="setHistoryViewMode('chart')" title="Chart View">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
                            <line x1="18" y1="20" x2="18" y2="10"></line>
                            <line x1="12" y1="20" x2="12" y2="4"></line>
                            <line x1="6" y1="20" x2="6" y2="14"></line>
                        </svg>
                    </button>
                </div>
            `;

            const history = historyData[selectedHistoryCamera] || [];
            const camera = cameras[selectedHistoryCamera];
            const unit = camera?.unit || '';

            let contentHtml = '';
            if (history.length > 0) {
                if (historyViewMode === 'table') {
                    contentHtml = renderHistoryTable(history, unit);
                } else if (historyViewMode === 'card') {
                    contentHtml = renderHistoryCards(history, unit);
                } else if (historyViewMode === 'chart') {
                    contentHtml = renderHistoryChart(history, unit);
                }
            } else {
                contentHtml = '<p style="color: var(--text-3);">No readings yet for this camera</p>';
            }

            container.innerHTML = `
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 16px; flex-wrap: wrap;">
                    <div class="history-tabs" style="margin-bottom: 0;">${tabs}</div>
                    ${viewModeHtml}
                </div>
                ${contentHtml}
            `;

            // If chart mode, render the chart
            if (historyViewMode === 'chart' && history.length > 0) {
                renderHistoryChartCanvas(history, unit);
            }
        }

        function renderHistoryTable(history, unit) {
            const rows = history.slice().reverse().map((entry, index) => {
                const time = new Date(entry.timestamp * 1000).toLocaleString();
                const confidence = entry.confidence || 0;
                const isLowConfidence = confidence < 80;
                const valueClass = isLowConfidence ? 'low-confidence' : '';
                const valueDisplay = entry.error
                    ? `<span class="error-cell">${entry.error}</span>`
                    : `<span class="${valueClass}">${entry.value !== null ? entry.value : '--'} ${unit}</span>`;

                // Confidence bar
                const barClass = confidence >= 80 ? 'high' : confidence >= 50 ? 'medium' : 'low';
                const confidenceDisplay = entry.confidence
                    ? `<span class="confidence-text">${confidence.toFixed(0)}%</span><div class="confidence-bar"><div class="confidence-bar-fill ${barClass}" style="width: ${confidence}%"></div></div>`
                    : '--';

                // Thumbnail
                const thumbnailHtml = entry.roi_image_id
                    ? `<img class="history-thumbnail" src="api/history-image/${encodeURIComponent(selectedHistoryCamera)}/${entry.roi_image_id}" onclick="showHistoryDetail(${JSON.stringify(entry).replace(/"/g, '&quot;')}, '${unit}')" onerror="this.outerHTML='<div class=\\'history-thumbnail-placeholder\\'>N/A</div>'">`
                    : '<div class="history-thumbnail-placeholder">N/A</div>';

                return `
                    <tr onclick="showHistoryDetail(${JSON.stringify(entry).replace(/"/g, '&quot;')}, '${unit}')" style="cursor: pointer;">
                        <td class="order-cell">${index + 1}</td>
                        <td style="width: 60px;">${thumbnailHtml}</td>
                        <td>${time}</td>
                        <td class="value-cell">${valueDisplay}</td>
                        <td>${confidenceDisplay}</td>
                        <td>${entry.ocr_provider || 'tesseract'}</td>
                    </tr>
                `;
            }).join('');

            return `
                <table class="history-table">
                    <thead>
                        <tr>
                            <th class="order-header">#</th>
                            <th>ROI</th>
                            <th>Time</th>
                            <th class="value-header">Value</th>
                            <th>Confidence</th>
                            <th>Provider</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            `;
        }

        function renderHistoryCards(history, unit) {
            const cards = history.slice().reverse().map((entry, index) => {
                const time = new Date(entry.timestamp * 1000).toLocaleString();
                const confidence = entry.confidence || 0;
                const isLowConfidence = confidence < 80;
                const valueClass = isLowConfidence ? 'low-confidence' : '';
                const barClass = confidence >= 80 ? 'high' : confidence >= 50 ? 'medium' : 'low';

                const thumbnailHtml = entry.roi_image_id
                    ? `<img style="width: 100%; height: 80px; object-fit: cover; border-radius: 6px;" src="api/history-image/${encodeURIComponent(selectedHistoryCamera)}/${entry.roi_image_id}" onerror="this.style.display='none'">`
                    : '';

                return `
                    <div style="background: var(--bg-3); border-radius: 8px; padding: 12px; cursor: pointer;" onclick="showHistoryDetail(${JSON.stringify(entry).replace(/"/g, '&quot;')}, '${unit}')">
                        ${thumbnailHtml}
                        <div style="font-size: 24px; font-weight: 700; color: ${isLowConfidence ? 'var(--error)' : 'var(--primary)'}; margin: 8px 0;">
                            ${entry.value !== null ? entry.value : '--'} ${unit}
                        </div>
                        <div style="font-size: 12px; color: var(--text-3);">${time}</div>
                        <div style="margin-top: 8px;">
                            <div class="confidence-bar" style="width: 100%;"><div class="confidence-bar-fill ${barClass}" style="width: ${confidence}%"></div></div>
                            <div style="font-size: 11px; color: var(--text-3); margin-top: 4px;">${confidence.toFixed(0)}% • ${entry.ocr_provider || 'tesseract'}</div>
                        </div>
                    </div>
                `;
            }).join('');

            return `<div class="grid grid-3" style="gap: 12px;">${cards}</div>`;
        }

        function renderHistoryChart(history, unit) {
            return `
                <div style="background: var(--bg-3); border-radius: 8px; padding: 16px;">
                    <canvas id="history-chart" height="250"></canvas>
                </div>
            `;
        }

        function renderHistoryChartCanvas(history, unit) {
            const canvas = document.getElementById('history-chart');
            if (!canvas) return;

            const ctx = canvas.getContext('2d');
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width - 32;
            canvas.height = 250;

            const values = history.map(e => e.value).filter(v => v !== null);
            if (values.length === 0) return;

            const minVal = Math.min(...values);
            const maxVal = Math.max(...values);
            const range = maxVal - minVal || 1;
            const padding = { top: 30, right: 20, bottom: 40, left: 50 };
            const chartWidth = canvas.width - padding.left - padding.right;
            const chartHeight = canvas.height - padding.top - padding.bottom;

            // Clear canvas
            ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg-3').trim() || '#1a1a2e';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Draw grid lines
            ctx.strokeStyle = 'rgba(255,255,255,0.1)';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 5; i++) {
                const y = padding.top + (chartHeight / 5) * i;
                ctx.beginPath();
                ctx.moveTo(padding.left, y);
                ctx.lineTo(canvas.width - padding.right, y);
                ctx.stroke();
            }

            // Draw Y axis labels
            ctx.fillStyle = 'rgba(255,255,255,0.5)';
            ctx.font = '11px system-ui';
            ctx.textAlign = 'right';
            for (let i = 0; i <= 5; i++) {
                const val = maxVal - (range / 5) * i;
                const y = padding.top + (chartHeight / 5) * i;
                ctx.fillText(val.toFixed(1), padding.left - 8, y + 4);
            }

            // Draw line chart
            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = 2;
            ctx.beginPath();

            history.forEach((entry, i) => {
                if (entry.value === null) return;
                const x = padding.left + (i / (history.length - 1 || 1)) * chartWidth;
                const y = padding.top + chartHeight - ((entry.value - minVal) / range) * chartHeight;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();

            // Draw points
            ctx.fillStyle = '#3b82f6';
            history.forEach((entry, i) => {
                if (entry.value === null) return;
                const x = padding.left + (i / (history.length - 1 || 1)) * chartWidth;
                const y = padding.top + chartHeight - ((entry.value - minVal) / range) * chartHeight;
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, Math.PI * 2);
                ctx.fill();
            });

            // X axis label
            ctx.fillStyle = 'rgba(255,255,255,0.5)';
            ctx.textAlign = 'center';
            ctx.fillText('Time →', canvas.width / 2, canvas.height - 8);

            // Y axis label
            ctx.save();
            ctx.translate(12, canvas.height / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText(`Value (${unit})`, 0, 0);
            ctx.restore();
        }

        function setHistoryViewMode(mode) {
            historyViewMode = mode;
            renderHistory();
        }

        function showHistoryDetail(entry, unit) {
            const time = new Date(entry.timestamp * 1000).toLocaleString();
            const confidence = entry.confidence || 0;
            const barClass = confidence >= 80 ? 'high' : confidence >= 50 ? 'medium' : 'low';

            const imageHtml = entry.roi_image_id
                ? `<img class="history-detail-image" src="api/history-image/${encodeURIComponent(selectedHistoryCamera)}/${entry.roi_image_id}" onerror="this.style.display='none'">`
                : '';

            const dialog = document.createElement('div');
            dialog.className = 'history-detail-dialog';
            dialog.onclick = (e) => { if (e.target === dialog) dialog.remove(); };
            dialog.innerHTML = `
                <div class="history-detail-content">
                    <div class="history-detail-header">
                        <h3>Extraction Details</h3>
                        <button class="modal-close" onclick="this.closest('.history-detail-dialog').remove()">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
                                <line x1="18" y1="6" x2="6" y2="18"></line>
                                <line x1="6" y1="6" x2="18" y2="18"></line>
                            </svg>
                        </button>
                    </div>
                    <div class="history-detail-body">
                        ${imageHtml}
                        <div class="history-detail-grid">
                            <div class="history-detail-item">
                                <label>Value</label>
                                <div class="value" style="color: ${confidence < 80 ? 'var(--error)' : 'var(--primary)'};">
                                    ${entry.value !== null ? entry.value : '--'} ${unit}
                                </div>
                            </div>
                            <div class="history-detail-item">
                                <label>Date & Time</label>
                                <div class="value">${time}</div>
                            </div>
                            <div class="history-detail-item">
                                <label>OCR Provider</label>
                                <div class="value">${entry.ocr_provider || 'tesseract'}</div>
                            </div>
                            <div class="history-detail-item">
                                <label>Confidence</label>
                                <div class="value">
                                    ${confidence.toFixed(1)}%
                                    <div class="confidence-bar" style="margin-top: 4px;"><div class="confidence-bar-fill ${barClass}" style="width: ${confidence}%"></div></div>
                                </div>
                            </div>
                            <div class="history-detail-item">
                                <label>Raw Text</label>
                                <div class="value">${entry.raw_text || '--'}</div>
                            </div>
                            ${entry.video_description ? `
                            <div class="history-detail-item full-width">
                                <label>Video Description</label>
                                <div class="value" style="font-weight: normal; font-size: 13px;">${entry.video_description}</div>
                            </div>
                            ` : ''}
                            ${entry.error ? `
                            <div class="history-detail-item full-width">
                                <label>Error</label>
                                <div class="value" style="color: var(--error);">${entry.error}</div>
                            </div>
                            ` : ''}
                        </div>
                    </div>
                </div>
            `;
            document.body.appendChild(dialog);
        }

        function selectHistoryCamera(name) {
            selectedHistoryCamera = name;
            renderHistory();
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
            const cameraEntries = Object.entries(cameras);
            const valueEntries = Object.entries(values);

            // No cameras configured at all
            if (cameraEntries.length === 0) {
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

            // Cameras configured but no values extracted yet
            if (valueEntries.length === 0) {
                grid.innerHTML = `
                    <div class="empty-state">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"></circle>
                            <polyline points="12 6 12 12 16 14"></polyline>
                        </svg>
                        <h3>Waiting for first extraction</h3>
                        <p>${cameraEntries.length} camera${cameraEntries.length > 1 ? 's' : ''} configured. Values will appear after first scan.</p>
                        <div class="loading" style="margin-top: 16px;"></div>
                    </div>
                `;
                return;
            }

            // Show all cameras, using values if available
            grid.innerHTML = cameraEntries.map(([name, camera]) => {
                const data = values[name] || {};
                const hasValue = data.value !== undefined && data.value !== null;
                const hasError = data.error;
                const statusClass = hasError ? 'error' : (hasValue ? 'ok' : 'pending');
                const statusText = hasError ? 'Error' : (hasValue ? 'OK' : 'Waiting');
                const displayValue = hasValue ? data.value : '--';
                const time = data.timestamp ? new Date(data.timestamp * 1000).toLocaleTimeString() : '';
                const confidence = data.confidence || 0;
                const isLowConfidence = hasValue && confidence < 80 && !hasError;
                const valueClass = isLowConfidence ? 'low-confidence' : '';

                const description = data.video_description || '';
                return `
                    <div class="value-card">
                        <div class="value-card-header">
                            <div class="value-card-name">${name}</div>
                            <div class="value-card-status ${statusClass}">${statusText}</div>
                        </div>
                        <div class="value-card-value ${valueClass}">
                            ${displayValue}<span class="value-card-unit">${camera.unit || ''}</span>
                        </div>
                        <div class="value-card-meta">
                            ${hasError ? `<span style="color: var(--error);">${data.error}</span>` :
                              (hasValue ? `Confidence: ${confidence.toFixed(0)}% • ${time}` : 'Waiting for first scan...')}
                            ${isLowConfidence ? '<span class="low-confidence-text"> (Low)</span>' : ''}
                        </div>
                        ${description ? `<div class="value-card-description">${description}</div>` : ''}
                    </div>
                `;
            }).join('');
        }

        function maskPasswordInUrl(url) {
            // Mask password in URL like rtsp://user:password@host -> rtsp://user:****@host
            try {
                const urlPattern = /^((?:rtsp|http|https):\/\/)([^:]+):([^@]+)@(.+)$/i;
                const match = url.match(urlPattern);
                if (match) {
                    return `${match[1]}${match[2]}:****@${match[4]}`;
                }
                // Also handle URLs with password in query params
                return url.replace(/([?&]password=)[^&]+/gi, '$1****');
            } catch (e) {
                return url;
            }
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
                        <div class="camera-item-url">${maskPasswordInUrl(cam.stream_url)}</div>
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
            document.getElementById('camera-unit').value = '°C';
            document.getElementById('camera-min-value').value = '';
            document.getElementById('camera-max-value').value = '';
            document.getElementById('camera-preprocessing').value = 'auto';
            document.getElementById('camera-template').value = '';
            document.getElementById('camera-use-template').checked = false;
            // Reset URL builder fields
            document.getElementById('camera-protocol').value = 'rtsp';
            document.getElementById('camera-host').value = '';
            document.getElementById('camera-port').value = '';
            document.getElementById('camera-path').value = '';
            document.getElementById('url-preview-text').textContent = '';
            setUrlMode('full');
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
            document.getElementById('camera-min-value').value = cam.min_value !== null && cam.min_value !== undefined ? cam.min_value : '';
            document.getElementById('camera-max-value').value = cam.max_value !== null && cam.max_value !== undefined ? cam.max_value : '';
            document.getElementById('camera-preprocessing').value = cam.preprocessing || 'auto';
            document.getElementById('camera-template').value = cam.template_name || '';
            document.getElementById('camera-use-template').checked = cam.use_template_matching || false;
            // Parse existing URL to builder fields
            setUrlMode('full');
            parseUrlToFields();
            document.getElementById('camera-modal').classList.add('active');
        }

        function closeCameraModal() {
            document.getElementById('camera-modal').classList.remove('active');
        }

        // URL Mode Toggle
        let urlMode = 'full';
        function setUrlMode(mode) {
            urlMode = mode;
            document.getElementById('url-mode-full').classList.toggle('active', mode === 'full');
            document.getElementById('url-mode-build').classList.toggle('active', mode === 'build');
            document.getElementById('url-input-full').style.display = mode === 'full' ? 'block' : 'none';
            document.getElementById('url-input-build').style.display = mode === 'build' ? 'block' : 'none';
            document.getElementById('url-preview').style.display = mode === 'build' ? 'block' : 'none';

            if (mode === 'build') {
                parseUrlToFields();
                buildUrlFromFields();
            }
        }

        // Parse full URL to individual fields
        function parseUrlToFields() {
            const urlInput = document.getElementById('camera-url').value.trim();
            if (!urlInput) return;

            try {
                // Handle URLs with credentials in format: protocol://user:pass@host:port/path
                let protocol = 'rtsp';
                let host = '';
                let port = '';
                let path = '';
                let username = '';
                let password = '';

                // Extract protocol
                const protocolMatch = urlInput.match(/^(rtsp|https?):\/\//i);
                if (protocolMatch) {
                    protocol = protocolMatch[1].toLowerCase();
                }

                // Remove protocol for parsing
                let remainder = urlInput.replace(/^(rtsp|https?):\/\//i, '');

                // Extract credentials if present (user:pass@)
                const credMatch = remainder.match(/^([^:@]+):([^@]+)@/);
                if (credMatch) {
                    username = decodeURIComponent(credMatch[1]);
                    password = decodeURIComponent(credMatch[2]);
                    remainder = remainder.substring(credMatch[0].length);
                } else {
                    // Check for username only (user@)
                    const userMatch = remainder.match(/^([^:@]+)@/);
                    if (userMatch) {
                        username = decodeURIComponent(userMatch[1]);
                        remainder = remainder.substring(userMatch[0].length);
                    }
                }

                // Extract host:port/path
                const hostMatch = remainder.match(/^([^:\/]+)(?::(\d+))?(\/.*)?$/);
                if (hostMatch) {
                    host = hostMatch[1];
                    port = hostMatch[2] || '';
                    path = hostMatch[3] || '';
                }

                // Update fields
                document.getElementById('camera-protocol').value = protocol;
                document.getElementById('camera-host').value = host;
                document.getElementById('camera-port').value = port;
                document.getElementById('camera-path').value = path;

                // Only update username/password if extracted and fields are empty
                if (username && !document.getElementById('camera-username').value) {
                    document.getElementById('camera-username').value = username;
                }
                if (password && !document.getElementById('camera-password').value) {
                    document.getElementById('camera-password').value = password;
                }
            } catch (e) {
                console.error('Failed to parse URL:', e);
            }
        }

        // Build URL from individual fields
        function buildUrlFromFields() {
            const protocol = document.getElementById('camera-protocol').value;
            const host = document.getElementById('camera-host').value.trim();
            const port = document.getElementById('camera-port').value.trim();
            let path = document.getElementById('camera-path').value.trim();
            const username = document.getElementById('camera-username').value.trim();
            const password = document.getElementById('camera-password').value;

            if (!host) {
                document.getElementById('url-preview-text').textContent = 'Enter host/IP to generate URL';
                return;
            }

            // Ensure path starts with /
            if (path && !path.startsWith('/')) {
                path = '/' + path;
            }

            // Build URL
            let url = protocol + '://';

            // Add credentials if provided
            if (username) {
                url += encodeURIComponent(username);
                if (password) {
                    url += ':' + encodeURIComponent(password);
                }
                url += '@';
            }

            url += host;

            // Add port if specified
            if (port) {
                url += ':' + port;
            }

            url += path;

            // Update the main URL field and preview
            document.getElementById('camera-url').value = url;
            document.getElementById('url-preview-text').textContent = url;
        }

        async function saveCamera() {
            const editName = document.getElementById('camera-edit-name').value;
            const name = document.getElementById('camera-name').value.trim();
            const url = document.getElementById('camera-url').value.trim();

            if (!name || !url) {
                toast('Please fill in required fields', 'error');
                return;
            }

            const minValInput = document.getElementById('camera-min-value').value;
            const maxValInput = document.getElementById('camera-max-value').value;

            const data = {
                name: name,
                stream_url: url,
                username: document.getElementById('camera-username').value,
                password: document.getElementById('camera-password').value,
                value_name: document.getElementById('camera-value-name').value || 'Value',
                unit: document.getElementById('camera-unit').value,
                min_value: minValInput !== '' ? parseFloat(minValInput) : null,
                max_value: maxValInput !== '' ? parseFloat(maxValInput) : null,
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
            const cameraNames = Object.keys(cameras);
            select.innerHTML = '<option value="">-- Select a camera --</option>' +
                cameraNames.map(name => `<option value="${name}">${name}</option>`).join('');

            // Auto-select first camera if available and none selected
            if (cameraNames.length > 0 && !select.value) {
                select.value = cameraNames[0];
                loadLivePreview();
            }
        }

        function editSelectedCamera() {
            const select = document.getElementById('live-camera-select');
            const cameraName = select.value;
            if (!cameraName) {
                toast('Please select a camera first', 'error');
                return;
            }
            // Open edit dialog for the selected camera
            editCamera(cameraName);
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
            loadSavedROIs();
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
                    // Reset zoom for new image
                    zoomLevel = 1;
                    setupCanvasSize();
                    drawROI();
                };

            } catch (e) {
                placeholder.innerHTML = `<p style="color: var(--error);">Failed to load preview</p>`;
            }
        }

        // Canvas & ROI
        let dragMode = 'none'; // 'none', 'draw', 'move', 'resize-tl', 'resize-tr', 'resize-bl', 'resize-br'
        let dragStartROI = null;
        const HANDLE_SIZE = 12; // Size of corner handles in image coordinates

        function getMousePos(canvas, e) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const x = (e.clientX - rect.left) * scaleX;
            const y = (e.clientY - rect.top) * scaleY;
            return {
                x: x / zoomLevel,
                y: y / zoomLevel
            };
        }

        function getHitTarget(pos) {
            if (currentROI.width === 0 || currentROI.height === 0) return 'draw';

            const roi = currentROI;
            const hs = HANDLE_SIZE / zoomLevel; // Adjust handle size for zoom

            // Check corner handles first
            if (Math.abs(pos.x - roi.x) < hs && Math.abs(pos.y - roi.y) < hs) return 'resize-tl';
            if (Math.abs(pos.x - (roi.x + roi.width)) < hs && Math.abs(pos.y - roi.y) < hs) return 'resize-tr';
            if (Math.abs(pos.x - roi.x) < hs && Math.abs(pos.y - (roi.y + roi.height)) < hs) return 'resize-bl';
            if (Math.abs(pos.x - (roi.x + roi.width)) < hs && Math.abs(pos.y - (roi.y + roi.height)) < hs) return 'resize-br';

            // Check if inside ROI (for moving)
            if (pos.x >= roi.x && pos.x <= roi.x + roi.width &&
                pos.y >= roi.y && pos.y <= roi.y + roi.height) {
                return 'move';
            }

            return 'draw';
        }

        function updateCursor(canvas, pos) {
            if (!previewImage || currentROI.width === 0) {
                canvas.style.cursor = 'crosshair';
                return;
            }

            const target = getHitTarget(pos);
            switch (target) {
                case 'resize-tl':
                case 'resize-br':
                    canvas.style.cursor = 'nwse-resize';
                    break;
                case 'resize-tr':
                case 'resize-bl':
                    canvas.style.cursor = 'nesw-resize';
                    break;
                case 'move':
                    canvas.style.cursor = 'move';
                    break;
                default:
                    canvas.style.cursor = 'crosshair';
            }
        }

        function setupCanvas() {
            const canvas = document.getElementById('preview-canvas');

            canvas.addEventListener('mousedown', (e) => {
                if (!previewImage) return;
                e.preventDefault();
                const pos = getMousePos(canvas, e);

                dragMode = getHitTarget(pos);
                isDrawing = true;
                startX = pos.x;
                startY = pos.y;
                dragStartROI = { ...currentROI };
            });

            canvas.addEventListener('mousemove', (e) => {
                if (!previewImage) return;
                e.preventDefault();
                const pos = getMousePos(canvas, e);

                // Update cursor based on position
                if (!isDrawing) {
                    updateCursor(canvas, pos);
                    return;
                }

                const dx = pos.x - startX;
                const dy = pos.y - startY;

                switch (dragMode) {
                    case 'move':
                        currentROI = {
                            x: Math.max(0, Math.round(dragStartROI.x + dx)),
                            y: Math.max(0, Math.round(dragStartROI.y + dy)),
                            width: dragStartROI.width,
                            height: dragStartROI.height
                        };
                        break;

                    case 'resize-tl':
                        currentROI = {
                            x: Math.round(Math.min(dragStartROI.x + dragStartROI.width - 10, dragStartROI.x + dx)),
                            y: Math.round(Math.min(dragStartROI.y + dragStartROI.height - 10, dragStartROI.y + dy)),
                            width: Math.max(10, Math.round(dragStartROI.width - dx)),
                            height: Math.max(10, Math.round(dragStartROI.height - dy))
                        };
                        break;

                    case 'resize-tr':
                        currentROI = {
                            x: dragStartROI.x,
                            y: Math.round(Math.min(dragStartROI.y + dragStartROI.height - 10, dragStartROI.y + dy)),
                            width: Math.max(10, Math.round(dragStartROI.width + dx)),
                            height: Math.max(10, Math.round(dragStartROI.height - dy))
                        };
                        break;

                    case 'resize-bl':
                        currentROI = {
                            x: Math.round(Math.min(dragStartROI.x + dragStartROI.width - 10, dragStartROI.x + dx)),
                            y: dragStartROI.y,
                            width: Math.max(10, Math.round(dragStartROI.width - dx)),
                            height: Math.max(10, Math.round(dragStartROI.height + dy))
                        };
                        break;

                    case 'resize-br':
                        currentROI = {
                            x: dragStartROI.x,
                            y: dragStartROI.y,
                            width: Math.max(10, Math.round(dragStartROI.width + dx)),
                            height: Math.max(10, Math.round(dragStartROI.height + dy))
                        };
                        break;

                    case 'draw':
                    default:
                        currentROI = {
                            x: Math.round(Math.min(startX, pos.x)),
                            y: Math.round(Math.min(startY, pos.y)),
                            width: Math.round(Math.abs(pos.x - startX)),
                            height: Math.round(Math.abs(pos.y - startY))
                        };
                        break;
                }

                updateROIDisplay();
                drawROI();
            });

            canvas.addEventListener('mouseup', () => {
                isDrawing = false;
                dragMode = 'none';
            });

            canvas.addEventListener('mouseleave', () => {
                isDrawing = false;
                dragMode = 'none';
            });
        }

        function setupCanvasSize() {
            const canvas = document.getElementById('preview-canvas');
            const img = document.getElementById('preview-image');

            // Set canvas to match zoomed image size
            const width = previewImage.width * zoomLevel;
            const height = previewImage.height * zoomLevel;

            canvas.width = width;
            canvas.height = height;
            canvas.style.width = width + 'px';
            canvas.style.height = height + 'px';
            canvas.style.display = 'block';

            // Set image size
            img.style.width = width + 'px';
            img.style.height = height + 'px';

            // Show zoom controls
            document.getElementById('zoom-controls').style.display = 'flex';
            document.getElementById('zoom-level').textContent = Math.round(zoomLevel * 100) + '%';

            imageScale = zoomLevel;
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

            // Show loading state
            const resultValue = document.getElementById('result-value');
            const resultRaw = document.getElementById('result-raw');
            const resultConfidence = document.getElementById('result-confidence');
            const resultPreview = document.getElementById('result-roi-preview');

            resultValue.textContent = '...';
            resultValue.style.color = 'var(--text-3)';
            resultRaw.innerHTML = '<div class="loading" style="width: 20px; height: 20px; margin: 0 auto;"></div>';
            resultConfidence.textContent = 'Extracting...';
            resultPreview.style.display = 'none';

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
                    // Show ROI preview if available
                    const previewEl = document.getElementById('result-roi-preview');
                    if (data.roi_preview) {
                        previewEl.innerHTML = `<img src="data:image/png;base64,${data.roi_preview}" style="width: 100%; display: block;">`;
                        previewEl.style.display = 'block';
                    } else {
                        previewEl.style.display = 'none';
                    }

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
                    document.getElementById('result-roi-preview').style.display = 'none';
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

        // Saved ROIs
        async function loadSavedROIs() {
            const name = document.getElementById('live-camera-select').value;
            if (!name) return;

            const section = document.getElementById('saved-rois-section');
            const list = document.getElementById('saved-rois-list');

            try {
                const res = await fetch(`api/saved-rois/${encodeURIComponent(name)}`);
                const rois = await res.json();

                if (rois.length > 0) {
                    section.style.display = 'block';
                    const validatedCount = rois.filter(r => r.validated_value !== undefined).length;
                    list.innerHTML = rois.map(roi => {
                        const isValidated = roi.validated_value !== undefined;
                        return `
                        <div class="saved-roi-item ${isValidated ? 'validated' : ''}" data-roi-id="${roi.id}" onclick="showSavedRoiDetail('${roi.id}', ${JSON.stringify(roi).replace(/"/g, '&quot;')})">
                            <img src="api/saved-rois/${encodeURIComponent(name)}/${roi.id}/image" alt="ROI">
                            ${isValidated ? '<div class="validated-badge" title="Validated for OCR training">&#x2714;</div>' : ''}
                            <div class="saved-roi-buttons">
                                <button class="saved-roi-apply" onclick="event.stopPropagation(); applySavedROI(${JSON.stringify(roi.roi).replace(/"/g, '&quot;')})" title="Apply">&#x2714;</button>
                                <button class="saved-roi-test" onclick="testSavedROI(${JSON.stringify(roi.roi).replace(/"/g, '&quot;')}, '${roi.id}', event)" title="Test">&#x25B6;</button>
                            </div>
                            <button class="saved-roi-delete" onclick="event.stopPropagation(); deleteSavedROI('${name}', '${roi.id}')">&times;</button>
                            <div class="saved-roi-info">
                                <div class="saved-roi-value" id="roi-value-${roi.id}">${isValidated ? roi.validated_value : (roi.extracted_value !== undefined && roi.extracted_value !== null ? roi.extracted_value : '--')}</div>
                                <div class="saved-roi-time">${new Date(roi.timestamp * 1000).toLocaleString()}</div>
                            </div>
                        </div>
                    `}).join('');
                } else {
                    section.style.display = 'block';
                    list.innerHTML = '<p style="color: var(--text-3); font-size: 12px;">No saved ROIs. Draw an ROI and click "Save Current ROI".</p>';
                }
            } catch (e) {
                console.error('Failed to load saved ROIs:', e);
            }
        }

        async function saveCurrentROI() {
            const name = document.getElementById('live-camera-select').value;
            if (!name) {
                toast('Please select a camera first', 'error');
                return;
            }

            if (currentROI.width <= 0 || currentROI.height <= 0) {
                toast('Please draw an ROI first', 'error');
                return;
            }

            // Get the current frame as base64 with ROI drawn
            const canvas = document.getElementById('preview-canvas');
            const img = document.getElementById('preview-image');

            // Create a temporary canvas to draw ROI cropped region
            const tempCanvas = document.createElement('canvas');
            const ctx = tempCanvas.getContext('2d');

            // Calculate actual ROI coordinates (accounting for zoom)
            const roiX = Math.round(currentROI.x / zoomLevel);
            const roiY = Math.round(currentROI.y / zoomLevel);
            const roiW = Math.round(currentROI.width / zoomLevel);
            const roiH = Math.round(currentROI.height / zoomLevel);

            // Draw the ROI region
            tempCanvas.width = roiW;
            tempCanvas.height = roiH;
            ctx.drawImage(img, roiX, roiY, roiW, roiH, 0, 0, roiW, roiH);

            const screenshot = tempCanvas.toDataURL('image/png').split(',')[1];

            // Also do a test extraction to get the value
            let extractedValue = null;
            try {
                const testRes = await fetch('api/test-extraction', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        camera_name: name,
                        roi: { x: roiX, y: roiY, width: roiW, height: roiH },
                        preprocessing: document.getElementById('live-preprocessing').value
                    })
                });
                const testData = await testRes.json();
                if (testData.success) {
                    extractedValue = testData.value;
                }
            } catch (e) {
                console.error('Test extraction failed:', e);
            }

            try {
                const res = await fetch(`api/saved-rois/${encodeURIComponent(name)}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        roi: { x: roiX, y: roiY, width: roiW, height: roiH },
                        screenshot: screenshot,
                        extracted_value: extractedValue
                    })
                });

                if (res.ok) {
                    toast('ROI saved', 'success');
                    loadSavedROIs();
                } else {
                    toast('Failed to save ROI', 'error');
                }
            } catch (e) {
                toast('Failed to save ROI', 'error');
            }
        }

        async function deleteSavedROI(cameraName, roiId) {
            if (!confirm('Delete this saved ROI?')) return;

            try {
                const res = await fetch(`api/saved-rois/${encodeURIComponent(cameraName)}/${roiId}`, {
                    method: 'DELETE'
                });

                if (res.ok) {
                    toast('ROI deleted', 'success');
                    loadSavedROIs();
                } else {
                    toast('Failed to delete ROI', 'error');
                }
            } catch (e) {
                toast('Failed to delete ROI', 'error');
            }
        }

        function applySavedROI(roi) {
            // Apply the saved ROI coordinates
            currentROI = {
                x: roi.x * zoomLevel,
                y: roi.y * zoomLevel,
                width: roi.width * zoomLevel,
                height: roi.height * zoomLevel
            };
            updateROIDisplay();
            drawROI();
            toast('ROI applied', 'success');
        }

        async function testAllROIs() {
            const name = document.getElementById('live-camera-select').value;
            if (!name) {
                toast('Please select a camera first', 'error');
                return;
            }

            const btn = document.getElementById('test-all-btn');
            btn.disabled = true;
            btn.textContent = 'Testing...';

            try {
                const res = await fetch(`api/saved-rois/${encodeURIComponent(name)}`);
                const rois = await res.json();

                if (rois.length === 0) {
                    toast('No saved ROIs to test', 'error');
                    btn.disabled = false;
                    btn.textContent = 'Test All';
                    return;
                }

                let bestResult = null;
                let bestConfidence = 0;

                for (const roi of rois) {
                    const valueEl = document.getElementById(`roi-value-${roi.id}`);
                    if (valueEl) {
                        valueEl.textContent = '...';
                        valueEl.style.color = 'var(--text-3)';
                    }

                    try {
                        const testRes = await fetch('api/test-extraction', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                camera_name: name,
                                roi: roi.roi,
                                preprocessing: document.getElementById('live-preprocessing').value
                            })
                        });

                        const data = await testRes.json();

                        if (valueEl) {
                            if (data.success) {
                                valueEl.textContent = data.value !== null ? data.value : '--';
                                valueEl.style.color = data.confidence > 50 ? 'var(--success)' : 'var(--warning)';

                                if (data.confidence > bestConfidence && data.value !== null) {
                                    bestConfidence = data.confidence;
                                    bestResult = { value: data.value, roi: roi.roi, id: roi.id };
                                }
                            } else {
                                valueEl.textContent = 'Error';
                                valueEl.style.color = 'var(--error)';
                            }
                        }
                    } catch (e) {
                        if (valueEl) {
                            valueEl.textContent = 'Error';
                            valueEl.style.color = 'var(--error)';
                        }
                    }
                }

                if (bestResult) {
                    toast(`Best: ${bestResult.value} (${bestConfidence.toFixed(0)}% confidence)`, 'success');
                    // Highlight best ROI
                    document.querySelectorAll('.saved-roi-item').forEach(el => el.style.borderColor = 'var(--border)');
                    const bestEl = document.querySelector(`[data-roi-id="${bestResult.id}"]`);
                    if (bestEl) bestEl.style.borderColor = 'var(--success)';
                } else {
                    toast('No valid results from any ROI', 'warning');
                }

            } catch (e) {
                toast('Test failed: ' + e.message, 'error');
            }

            btn.disabled = false;
            btn.textContent = 'Test All';
        }

        // Store last test results for ROI detail dialog
        let savedRoiTestResults = {};

        async function testSavedROI(roi, roiId, event) {
            if (event) event.stopPropagation();
            const name = document.getElementById('live-camera-select').value;
            if (!name) {
                toast('Please select a camera first', 'error');
                return;
            }

            // Show busy indicator on the card
            const cardEl = document.querySelector(`.saved-roi-item[data-roi-id="${roiId}"]`);
            const valueEl = document.getElementById(`roi-value-${roiId}`);
            if (cardEl) {
                cardEl.style.opacity = '0.6';
                cardEl.style.pointerEvents = 'none';
            }
            if (valueEl) {
                valueEl.innerHTML = '<div class="loading" style="width: 16px; height: 16px; margin: 0 auto;"></div>';
            }

            try {
                const res = await fetch('api/test-extraction', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        camera_name: name,
                        roi: roi,
                        preprocessing: document.getElementById('live-preprocessing').value
                    })
                });

                const data = await res.json();

                // Store result for detail dialog
                savedRoiTestResults[roiId] = {
                    ...data,
                    roi: roi,
                    timestamp: Date.now() / 1000,
                    camera_name: name
                };

                if (data.success && valueEl) {
                    valueEl.textContent = data.value !== null ? data.value : '--';
                    valueEl.style.color = data.confidence > 50 ? 'var(--success)' : 'var(--warning)';
                    toast(`Value: ${data.value} (${data.confidence.toFixed(0)}%)`, 'success');
                } else if (valueEl) {
                    valueEl.textContent = 'Error';
                    valueEl.style.color = 'var(--error)';
                }
            } catch (e) {
                if (valueEl) {
                    valueEl.textContent = 'Error';
                    valueEl.style.color = 'var(--error)';
                }
                toast('Test failed', 'error');
            } finally {
                // Restore card state
                if (cardEl) {
                    cardEl.style.opacity = '1';
                    cardEl.style.pointerEvents = 'auto';
                }
            }
        }

        function showSavedRoiDetail(roiId, roiData) {
            const name = document.getElementById('live-camera-select').value;
            const testResult = savedRoiTestResults[roiId] || {};
            const camera = cameras[name] || {};
            const unit = camera.unit || '';

            const time = roiData.timestamp ? new Date(roiData.timestamp * 1000).toLocaleString() : '--';
            const value = testResult.value !== undefined ? testResult.value : (roiData.extracted_value !== undefined ? roiData.extracted_value : '--');
            const confidence = testResult.confidence || roiData.confidence || 0;
            const barClass = confidence >= 80 ? 'high' : confidence >= 50 ? 'medium' : 'low';
            const rawText = testResult.raw_text || roiData.raw_text || '--';
            const validatedValue = roiData.validated_value;
            const validatedAt = roiData.validated_at ? new Date(roiData.validated_at * 1000).toLocaleString() : null;

            const dialog = document.createElement('div');
            dialog.className = 'history-detail-dialog';
            dialog.onclick = (e) => { if (e.target === dialog) dialog.remove(); };
            dialog.innerHTML = `
                <div class="history-detail-content">
                    <div class="history-detail-header">
                        <h3>Saved ROI Details</h3>
                        <button class="modal-close" onclick="this.closest('.history-detail-dialog').remove()">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
                                <line x1="18" y1="6" x2="6" y2="18"></line>
                                <line x1="6" y1="6" x2="18" y2="18"></line>
                            </svg>
                        </button>
                    </div>
                    <div class="history-detail-body">
                        <img class="history-detail-image" src="api/saved-rois/${encodeURIComponent(name)}/${roiId}/image" onerror="this.style.display='none'">
                        <div class="history-detail-grid">
                            <div class="history-detail-item">
                                <label>OCR Value</label>
                                <div class="value" style="color: ${confidence < 80 && confidence > 0 ? 'var(--error)' : 'var(--primary)'};">
                                    ${value} ${unit}
                                </div>
                            </div>
                            <div class="history-detail-item">
                                <label>Validated Value</label>
                                <div class="value" style="color: ${validatedValue !== undefined ? 'var(--success)' : 'var(--text-3)'};">
                                    ${validatedValue !== undefined ? validatedValue + ' ' + unit : 'Not validated'}
                                    ${validatedAt ? `<div style="font-size: 11px; color: var(--text-3);">${validatedAt}</div>` : ''}
                                </div>
                            </div>
                            <div class="history-detail-item">
                                <label>Saved On</label>
                                <div class="value">${time}</div>
                            </div>
                            <div class="history-detail-item">
                                <label>Confidence</label>
                                <div class="value">
                                    ${confidence > 0 ? confidence.toFixed(1) + '%' : '--'}
                                    ${confidence > 0 ? `<div class="confidence-bar" style="margin-top: 4px;"><div class="confidence-bar-fill ${barClass}" style="width: ${confidence}%"></div></div>` : ''}
                                </div>
                            </div>
                            <div class="history-detail-item">
                                <label>Raw Text</label>
                                <div class="value">${rawText}</div>
                            </div>
                            <div class="history-detail-item">
                                <label>ROI Coordinates</label>
                                <div class="value" style="font-family: monospace; font-size: 12px;">
                                    X: ${roiData.roi?.x || 0}, Y: ${roiData.roi?.y || 0}, W: ${roiData.roi?.width || 0}, H: ${roiData.roi?.height || 0}
                                </div>
                            </div>
                        </div>
                        <div style="margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap;">
                            <button class="btn btn-primary" onclick="applySavedROI(${JSON.stringify(roiData.roi)}); this.closest('.history-detail-dialog').remove();">Apply ROI</button>
                            <button class="btn btn-secondary" onclick="testSavedROI(${JSON.stringify(roiData.roi)}, '${roiId}'); this.closest('.history-detail-dialog').remove();">Test Extract</button>
                            <button class="btn btn-secondary" onclick="showValidateDialog('${roiId}', ${JSON.stringify(roiData).replace(/"/g, '&quot;')})" style="background: var(--warning); color: #000;">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16" style="margin-right: 4px;">
                                    <path d="M9 11l3 3L22 4"></path>
                                    <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"></path>
                                </svg>
                                Validate
                            </button>
                        </div>
                    </div>
                </div>
            `;
            document.body.appendChild(dialog);
        }

        function showValidateDialog(roiId, roiData) {
            const name = document.getElementById('live-camera-select').value;
            const camera = cameras[name] || {};
            const unit = camera.unit || '';
            const currentValue = roiData.validated_value !== undefined ? roiData.validated_value : (roiData.extracted_value || '');

            // Close the detail dialog first
            document.querySelectorAll('.history-detail-dialog').forEach(d => d.remove());

            const dialog = document.createElement('div');
            dialog.className = 'history-detail-dialog';
            dialog.onclick = (e) => { if (e.target === dialog) dialog.remove(); };
            dialog.innerHTML = `
                <div class="history-detail-content" style="max-width: 400px;">
                    <div class="history-detail-header">
                        <h3>Validate OCR Value</h3>
                        <button class="modal-close" onclick="this.closest('.history-detail-dialog').remove()">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
                                <line x1="18" y1="6" x2="6" y2="18"></line>
                                <line x1="6" y1="6" x2="18" y2="18"></line>
                            </svg>
                        </button>
                    </div>
                    <div class="history-detail-body">
                        <img class="history-detail-image" src="api/saved-rois/${encodeURIComponent(name)}/${roiId}/image" style="max-height: 150px;" onerror="this.style.display='none'">
                        <p style="margin: 16px 0 8px; color: var(--text-2);">
                            What is the correct value shown in this image?
                        </p>
                        <div class="form-group" style="margin-bottom: 16px;">
                            <input type="text" id="validate-value-input" class="form-control"
                                   value="${currentValue}" placeholder="Enter the correct numeric value"
                                   style="font-size: 18px; text-align: center;">
                            <small style="color: var(--text-3);">Unit: ${unit || 'none'}</small>
                        </div>
                        <p style="font-size: 12px; color: var(--text-3); margin-bottom: 16px;">
                            This helps train the OCR system to better recognize values from your camera.
                            After validating several ROIs, use "Train OCR" to find optimal settings.
                        </p>
                        <div style="display: flex; gap: 8px;">
                            <button class="btn btn-primary" onclick="submitValidation('${roiId}')" style="flex: 1;">
                                Save Validated Value
                            </button>
                            <button class="btn btn-secondary" onclick="this.closest('.history-detail-dialog').remove()">
                                Cancel
                            </button>
                        </div>
                    </div>
                </div>
            `;
            document.body.appendChild(dialog);

            // Focus the input
            setTimeout(() => document.getElementById('validate-value-input').focus(), 100);
        }

        async function submitValidation(roiId) {
            const name = document.getElementById('live-camera-select').value;
            const value = document.getElementById('validate-value-input').value.trim();

            if (!value) {
                toast('Please enter a value', 'error');
                return;
            }

            try {
                const res = await fetch(`api/saved-rois/${encodeURIComponent(name)}/${roiId}/validate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ value: value })
                });

                if (res.ok) {
                    toast('Value validated successfully', 'success');
                    document.querySelectorAll('.history-detail-dialog').forEach(d => d.remove());
                    loadSavedROIs();
                } else {
                    const err = await res.json();
                    toast(err.error || 'Failed to validate', 'error');
                }
            } catch (e) {
                toast('Failed to validate value', 'error');
            }
        }

        async function trainOCR() {
            const name = document.getElementById('live-camera-select').value;
            if (!name) {
                toast('Please select a camera first', 'error');
                return;
            }

            const btn = event.target;
            btn.disabled = true;
            btn.innerHTML = '<div class="loading" style="width: 16px; height: 16px;"></div> Training...';

            try {
                const res = await fetch(`api/saved-rois/${encodeURIComponent(name)}/train`, {
                    method: 'POST'
                });

                const data = await res.json();

                if (data.error) {
                    toast(data.error, 'error');
                } else {
                    // Show training results
                    showTrainingResults(data);
                }
            } catch (e) {
                toast('Training failed: ' + e.message, 'error');
            }

            btn.disabled = false;
            btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16" style="margin-right: 4px;">
                <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path>
                <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path>
            </svg> Train OCR`;
        }

        function showTrainingResults(data) {
            const dialog = document.createElement('div');
            dialog.className = 'history-detail-dialog';
            dialog.onclick = (e) => { if (e.target === dialog) dialog.remove(); };

            const best = data.best_config || {};
            const ranked = data.ranked_configs || [];

            dialog.innerHTML = `
                <div class="history-detail-content" style="max-width: 500px;">
                    <div class="history-detail-header">
                        <h3>OCR Training Results</h3>
                        <button class="modal-close" onclick="this.closest('.history-detail-dialog').remove()">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
                                <line x1="18" y1="6" x2="6" y2="18"></line>
                                <line x1="6" y1="6" x2="18" y2="18"></line>
                            </svg>
                        </button>
                    </div>
                    <div class="history-detail-body">
                        <div style="background: var(--bg); padding: 16px; border-radius: 8px; margin-bottom: 16px;">
                            <h4 style="margin: 0 0 8px; color: var(--success);">Best Configuration</h4>
                            <p style="margin: 4px 0; font-size: 14px;">
                                <strong>Preprocessing:</strong> ${best.preprocessing || 'auto'}<br>
                                <strong>PSM Mode:</strong> ${best.psm || 7}<br>
                                <strong>Accuracy:</strong> ${(best.accuracy || 0).toFixed(1)}% (${best.matches || 0}/${best.total || 0} matches)
                            </p>
                        </div>

                        <p style="margin: 0 0 8px; font-weight: 600;">Tested on ${data.validated_count || 0} validated ROI(s)</p>

                        <div style="margin-top: 16px;">
                            <h4 style="margin: 0 0 8px;">Top Configurations</h4>
                            <table style="width: 100%; font-size: 12px; border-collapse: collapse;">
                                <thead>
                                    <tr style="background: var(--bg);">
                                        <th style="padding: 8px; text-align: left;">Preprocessing</th>
                                        <th style="padding: 8px; text-align: center;">PSM</th>
                                        <th style="padding: 8px; text-align: right;">Accuracy</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${ranked.map((cfg, i) => `
                                        <tr style="${i === 0 ? 'background: rgba(76, 175, 80, 0.1);' : ''}">
                                            <td style="padding: 8px;">${cfg.preprocessing}</td>
                                            <td style="padding: 8px; text-align: center;">${cfg.psm}</td>
                                            <td style="padding: 8px; text-align: right;">${(cfg.matches / cfg.total * 100).toFixed(1)}%</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>

                        <p style="font-size: 12px; color: var(--text-3); margin-top: 16px;">
                            To use the best configuration, set the camera's preprocessing to "${best.preprocessing || 'auto'}".
                        </p>

                        <div style="margin-top: 16px;">
                            <button class="btn btn-primary" onclick="this.closest('.history-detail-dialog').remove()">Close</button>
                        </div>
                    </div>
                </div>
            `;
            document.body.appendChild(dialog);
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
                            <p>No cameras were discovered on your network.<br>Check that cameras are powered on and connected.</p>
                        </div>
                    `;
                } else {
                    list.innerHTML = discovered.map((cam, idx) => {
                        const safeName = (cam.name || cam.ip).replace(/'/g, "\\'");
                        const safeMfr = (cam.manufacturer || '').replace(/'/g, "\\'");
                        const onvifInfo = cam.onvif_port ? ` • ONVIF:${cam.onvif_port}` : '';
                        return `
                        <div class="discovery-item" id="discovery-${idx}">
                            <div class="discovery-item-header">
                                <div class="discovery-item-info">
                                    <h4>${cam.name || cam.ip}</h4>
                                    <p>${cam.manufacturer ? cam.manufacturer + ' • ' : ''}${cam.ip}:${cam.port}${onvifInfo}</p>
                                </div>
                            </div>
                            <div class="discovery-item-preview" id="discovery-preview-${idx}">
                                <div class="preview-placeholder">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                                        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                                        <circle cx="12" cy="13" r="4"></circle>
                                    </svg>
                                    <div class="loading" style="margin-top: 8px;"></div>
                                    <p style="margin-top: 8px;">Loading preview...</p>
                                </div>
                            </div>
                            <div class="discovery-item-actions">
                                <button class="btn btn-secondary btn-sm" onclick="previewDiscoveredCamera(${idx}, '${cam.ip}', ${cam.port}, '${cam.stream_url}')">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
                                        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                                        <circle cx="12" cy="12" r="3"></circle>
                                    </svg>
                                    Preview
                                </button>
                                <button class="btn btn-primary btn-sm" onclick="addDiscoveredCamera('${cam.ip}', ${cam.port}, '${cam.stream_url}', '${safeMfr}', '${safeName}')">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
                                        <line x1="12" y1="5" x2="12" y2="19"></line>
                                        <line x1="5" y1="12" x2="19" y2="12"></line>
                                    </svg>
                                    Add
                                </button>
                            </div>
                        </div>`;
                    }).join('');

                    // Auto-load previews for discovered cameras
                    discovered.forEach((cam, idx) => {
                        loadDiscoveryPreview(idx, cam.ip, cam.port, cam.stream_url);
                    });
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

        async function loadDiscoveryPreview(idx, ip, port, streamUrl) {
            const previewEl = document.getElementById(`discovery-preview-${idx}`);
            if (!previewEl) return;

            try {
                // Try to capture a frame using the test endpoint
                const testUrl = streamUrl.replace('{user}', 'admin').replace('{pass}', 'admin');
                const res = await fetch('api/test-capture', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url: testUrl, username: '', password: '' })
                });

                if (res.ok) {
                    const data = await res.json();
                    if (data.frame) {
                        previewEl.innerHTML = `<img src="data:image/png;base64,${data.frame}" alt="Preview">`;
                        return;
                    }
                }
                previewEl.innerHTML = `<div class="preview-error">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                        <circle cx="12" cy="13" r="4"></circle>
                    </svg>
                    <span>Credentials required</span>
                    <small>Add camera and configure username/password</small>
                </div>`;
            } catch (e) {
                previewEl.innerHTML = `<div class="preview-error">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                        <circle cx="12" cy="13" r="4"></circle>
                    </svg>
                    <span>Credentials required</span>
                    <small>Add camera and configure username/password</small>
                </div>`;
            }
        }

        function previewDiscoveredCamera(idx, ip, port, streamUrl) {
            const previewEl = document.getElementById(`discovery-preview-${idx}`);
            if (!previewEl) return;

            previewEl.innerHTML = '<div class="preview-placeholder"><div class="loading"></div><p style="margin-top: 8px;">Loading preview...</p></div>';
            loadDiscoveryPreview(idx, ip, port, streamUrl);
        }

        function addDiscoveredCamera(ip, port, streamUrl, manufacturer, name) {
            // Open the modal first
            openAddCameraModal();
            document.getElementById('camera-modal-title').textContent = 'Add Discovered Camera';

            // Set camera name
            const cameraName = name || (manufacturer ? `${manufacturer} ${ip}` : `Camera ${ip}`);
            document.getElementById('camera-name').value = cameraName;

            // Set the full URL with placeholder credentials
            const fullUrl = streamUrl.replace('{user}', 'admin').replace('{pass}', 'admin');
            document.getElementById('camera-url').value = fullUrl;

            // Parse URL to populate builder fields
            parseUrlToFields();

            // Clear password since it's just a placeholder
            document.getElementById('camera-password').value = '';

            // Set default port if not in URL
            if (!document.getElementById('camera-port').value) {
                document.getElementById('camera-port').value = port || '554';
            }
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

        // AI Settings
        const AI_MODEL_HINTS = {
            'none': '',
            'openai': 'Default: gpt-4o (supports vision)',
            'anthropic': 'Default: claude-sonnet-4-20250514',
            'google': 'Default: gemini-1.5-flash',
            'ollama': 'Default: llava (requires vision model)',
            'custom': 'Enter model name from your provider',
            'google-vision': 'Uses TEXT_DETECTION (no model needed)',
            'google-docai': 'Requires processor path in URL field',
            'azure-ocr': 'Uses Read API v3.2 (no model needed)',
            'aws-textract': 'Uses detect-document-text (no model needed)'
        };

        async function loadAIConfig() {
            try {
                const res = await fetch('api/ai-config');
                const config = await res.json();

                document.getElementById('ai-provider').value = config.provider;
                document.getElementById('ai-url').value = config.api_url || '';
                document.getElementById('ai-model').value = config.model || '';
                document.getElementById('ai-region').value = config.region || '';
                document.getElementById('ai-enable-ocr').checked = config.enabled_for_ocr;
                document.getElementById('ai-enable-description').checked = config.enabled_for_description;

                onAIProviderChange();

                if (config.has_api_key) {
                    document.getElementById('ai-apikey').placeholder = '••••••••••••••••';
                }
            } catch (e) {
                console.error('Failed to load AI config:', e);
            }
        }

        function onAIProviderChange() {
            const provider = document.getElementById('ai-provider').value;
            const configFields = document.getElementById('ai-config-fields');
            const urlGroup = document.getElementById('ai-url-group');
            const urlLabel = document.getElementById('ai-url-label');
            const urlHint = document.getElementById('ai-url-hint');
            const apiKeyGroup = document.getElementById('ai-apikey-group');
            const regionGroup = document.getElementById('ai-region-group');
            const modelHint = document.getElementById('ai-model-hint');
            const modelGroup = document.getElementById('ai-model').parentElement;
            const descriptionCheckbox = document.getElementById('ai-enable-description').parentElement;
            const testSection = document.getElementById('ai-test-section');

            const cloudOcrProviders = ['google-vision', 'google-docai', 'azure-ocr', 'aws-textract'];
            const isCloudOcr = cloudOcrProviders.includes(provider);

            if (provider === 'none') {
                configFields.style.display = 'none';
                testSection.style.display = 'none';
            } else {
                configFields.style.display = 'block';
                testSection.style.display = 'block';

                // Show/hide URL field
                const needsUrl = provider === 'ollama' || provider === 'custom' || provider === 'azure-ocr' || provider === 'google-docai';
                urlGroup.style.display = needsUrl ? 'block' : 'none';

                // Update URL hint based on provider
                if (provider === 'ollama') {
                    urlLabel.textContent = 'API URL';
                    urlHint.textContent = 'URL to your Ollama server (e.g., http://localhost:11434)';
                } else if (provider === 'custom') {
                    urlLabel.textContent = 'API URL';
                    urlHint.textContent = 'Base URL of OpenAI-compatible API (e.g., http://localhost:1234/v1)';
                } else if (provider === 'azure-ocr') {
                    urlLabel.textContent = 'Endpoint URL';
                    urlHint.textContent = 'Azure endpoint (e.g., https://westus.api.cognitive.microsoft.com)';
                } else if (provider === 'google-docai') {
                    urlLabel.textContent = 'Processor Path';
                    urlHint.textContent = 'Format: projects/PROJECT_ID/locations/LOCATION/processors/PROCESSOR_ID';
                }

                // Show/hide region field for AWS
                regionGroup.style.display = provider === 'aws-textract' ? 'block' : 'none';

                // Show API key for all except Ollama
                apiKeyGroup.style.display = provider === 'ollama' ? 'none' : 'block';

                // Update API key hint for AWS
                const apiKeyHint = apiKeyGroup.querySelector('p');
                if (provider === 'aws-textract') {
                    apiKeyHint.textContent = 'Format: ACCESS_KEY_ID:SECRET_ACCESS_KEY';
                } else {
                    apiKeyHint.textContent = 'Your API key is stored locally and never shared.';
                }

                // Hide model field and description checkbox for cloud OCR services
                modelGroup.style.display = isCloudOcr ? 'none' : 'block';
                descriptionCheckbox.style.display = isCloudOcr ? 'none' : 'flex';

                // Update model hint
                modelHint.textContent = AI_MODEL_HINTS[provider] || 'Enter your model name';
            }
        }

        async function saveAIConfig() {
            const provider = document.getElementById('ai-provider').value;
            const apiKey = document.getElementById('ai-apikey').value;
            const apiUrl = document.getElementById('ai-url').value;
            const model = document.getElementById('ai-model').value;
            const region = document.getElementById('ai-region').value;
            const enableOcr = document.getElementById('ai-enable-ocr').checked;
            const enableDescription = document.getElementById('ai-enable-description').checked;

            try {
                const res = await fetch('api/ai-config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        provider,
                        api_key: apiKey,
                        api_url: apiUrl,
                        model,
                        region,
                        enabled_for_ocr: enableOcr,
                        enabled_for_description: enableDescription
                    })
                });

                if (res.ok) {
                    toast('AI configuration saved', 'success');
                } else {
                    toast('Failed to save configuration', 'error');
                }
            } catch (e) {
                toast('Error: ' + e.message, 'error');
            }
        }

        let aiTestImageBase64 = null;

        function onAITestSourceChange() {
            const source = document.getElementById('ai-test-source').value;
            document.getElementById('ai-test-upload-section').style.display = source === 'upload' ? 'block' : 'none';
            document.getElementById('ai-test-camera-section').style.display = source === 'camera' ? 'block' : 'none';
            aiTestImageBase64 = null;

            if (source === 'camera') {
                // Populate camera dropdown
                const select = document.getElementById('ai-test-camera');
                const cameraNames = Object.keys(cameras);
                select.innerHTML = '<option value="">-- Select Camera --</option>' +
                    cameraNames.map(name => `<option value="${name}">${name}</option>`).join('');
            }
        }

        function onAITestFileSelect() {
            const file = document.getElementById('ai-test-file').files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(e) {
                aiTestImageBase64 = e.target.result.split(',')[1];  // Remove data:image/...;base64, prefix
                document.getElementById('ai-test-preview-img').src = e.target.result;
                document.getElementById('ai-test-preview-upload').style.display = 'block';
            };
            reader.readAsDataURL(file);
        }

        async function onAITestCameraChange() {
            const cameraName = document.getElementById('ai-test-camera').value;
            if (!cameraName) {
                document.getElementById('ai-test-preview-camera').style.display = 'none';
                aiTestImageBase64 = null;
                return;
            }

            // Fetch camera preview with ROI
            try {
                const res = await fetch(`api/preview/${encodeURIComponent(cameraName)}?roi=true`);
                if (res.ok) {
                    const data = await res.json();
                    if (data.frame) {
                        aiTestImageBase64 = data.roi_frame || data.frame;
                        document.getElementById('ai-test-camera-img').src = 'data:image/png;base64,' + data.frame;
                        document.getElementById('ai-test-preview-camera').style.display = 'block';
                    }
                }
            } catch (e) {
                toast('Failed to load camera preview', 'error');
            }
        }

        async function testAIProvider() {
            const testBtn = document.getElementById('test-ai-btn');
            const resultsCard = document.getElementById('ai-test-results');
            const resultsContent = document.getElementById('ai-test-content');

            const source = document.getElementById('ai-test-source').value;
            let imageBase64 = aiTestImageBase64;
            let cameraName = null;

            if (source === 'camera') {
                cameraName = document.getElementById('ai-test-camera').value;
                if (!cameraName) {
                    toast('Please select a camera', 'error');
                    return;
                }
            } else if (!imageBase64) {
                toast('Please select an image first', 'error');
                return;
            }

            testBtn.disabled = true;
            testBtn.innerHTML = '<div class="loading" style="width: 16px; height: 16px;"></div> Testing...';
            resultsCard.style.display = 'block';
            resultsContent.innerHTML = '<div class="loading"></div><p style="text-align: center; margin-top: 12px;">Testing AI provider...</p>';

            try {
                const payload = imageBase64 ? { image: imageBase64 } : { camera: cameraName };
                const res = await fetch('api/ai-test', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await res.json();

                if (data.error) {
                    resultsContent.innerHTML = `<p style="color: var(--error);">Error: ${data.error}</p>`;
                } else {
                    let html = '<div style="display: flex; flex-direction: column; gap: 12px;">';

                    if (data.description) {
                        html += `
                            <div>
                                <strong>Scene Description:</strong>
                                <p style="background: var(--bg); padding: 12px; border-radius: 6px; margin-top: 4px;">${data.description}</p>
                            </div>
                        `;
                    }

                    if (data.ocr) {
                        html += `
                            <div>
                                <strong>OCR Result:</strong>
                                <p style="background: var(--bg); padding: 12px; border-radius: 6px; margin-top: 4px;">${data.ocr}</p>
                            </div>
                        `;
                    }

                    if (!data.description && !data.ocr) {
                        html += '<p>No features enabled. Enable OCR enhancement or scene description above.</p>';
                    }

                    html += '</div>';
                    resultsContent.innerHTML = html;
                    toast('AI test completed', 'success');
                }
            } catch (e) {
                resultsContent.innerHTML = `<p style="color: var(--error);">Error: ${e.message}</p>`;
            }

            testBtn.disabled = false;
            testBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
                <polygon points="5 3 19 12 5 21 5 3"></polygon>
            </svg> Test AI Provider`;
        }

        // Load AI config when visiting AI settings page
        document.querySelector('[data-page="ai-settings"]').addEventListener('click', loadAIConfig);
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


@app.route('/api/history')
def get_history():
    """Get value history for all cameras."""
    return jsonify(processor.history)


@app.route('/api/history/<camera_name>')
def get_camera_history(camera_name):
    """Get value history for a specific camera."""
    history = processor.history.get(camera_name, [])
    return jsonify(history)


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
        'min_value': data.get('min_value'),
        'max_value': data.get('max_value'),
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
                'min_value': data.get('min_value', cam.min_value),
                'max_value': data.get('max_value', cam.max_value),
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

    # Draw ROI rectangle only (value shown in sidebar, not on image)
    if camera.roi_width > 0 and camera.roi_height > 0:
        frame = processor.draw_roi_on_frame(
            frame,
            {'x': camera.roi_x, 'y': camera.roi_y, 'width': camera.roi_width, 'height': camera.roi_height}
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

    # Add cropped ROI preview to result
    if roi.get('width', 0) > 0 and roi.get('height', 0) > 0:
        x, y, w, h = roi.get('x', 0), roi.get('y', 0), roi.get('width', 0), roi.get('height', 0)
        img_h, img_w = frame.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        roi_crop = frame[y:y+h, x:x+w]
        if roi_crop.size > 0:
            # Scale up if too small
            if roi_crop.shape[0] < 50 or roi_crop.shape[1] < 50:
                scale = max(100 / roi_crop.shape[0], 100 / roi_crop.shape[1], 2)
                roi_crop = cv2.resize(roi_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            _, buffer = cv2.imencode('.png', roi_crop)
            result['roi_preview'] = base64.b64encode(buffer).decode('utf-8')

    return jsonify(result)


@app.route('/api/test-capture', methods=['POST'])
def test_capture():
    """Test capture from a URL (for discovery preview)."""
    data = request.get_json()
    url = data.get('url', '')
    username = data.get('username', '')
    password = data.get('password', '')

    if not url:
        return jsonify({'success': False, 'error': 'URL is required'}), 400

    frame, error = processor.capture_frame_from_url(url, username, password)
    if error:
        return jsonify({'success': False, 'error': error})

    # Resize for thumbnail
    h, w = frame.shape[:2]
    max_size = 400
    if w > max_size or h > max_size:
        scale = max_size / max(w, h)
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    frame_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'success': True,
        'frame': frame_b64,
        'width': frame.shape[1],
        'height': frame.shape[0]
    })


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


@app.route('/api/saved-rois/<camera_name>', methods=['GET'])
def get_saved_rois(camera_name):
    """Get saved ROIs for a camera."""
    rois = []
    roi_dir = Path(SAVED_ROIS_PATH) / camera_name
    if roi_dir.exists():
        for json_file in roi_dir.glob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    roi_data = json.load(f)
                    roi_data['id'] = json_file.stem
                    rois.append(roi_data)
            except Exception as e:
                logger.error(f"Error loading ROI {json_file}: {e}")
    # Sort by timestamp descending
    rois.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
    return jsonify(rois)


@app.route('/api/saved-rois/<camera_name>', methods=['POST'])
def save_roi(camera_name):
    """Save an ROI with screenshot."""
    data = request.json
    roi = data.get('roi', {})
    screenshot = data.get('screenshot', '')  # base64 image
    extracted_value = data.get('extracted_value')

    if not roi or not screenshot:
        return jsonify({'error': 'ROI and screenshot required'}), 400

    # Create camera ROI directory
    roi_dir = Path(SAVED_ROIS_PATH) / camera_name
    roi_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique ID
    roi_id = f"roi_{int(time.time() * 1000)}"

    # Save screenshot
    img_data = base64.b64decode(screenshot)
    img_path = roi_dir / f"{roi_id}.png"
    with open(img_path, 'wb') as f:
        f.write(img_data)

    # Save ROI metadata
    roi_data = {
        'roi': roi,
        'timestamp': time.time(),
        'camera': camera_name,
        'extracted_value': extracted_value
    }
    json_path = roi_dir / f"{roi_id}.json"
    with open(json_path, 'w') as f:
        json.dump(roi_data, f)

    return jsonify({'success': True, 'id': roi_id})


@app.route('/api/saved-rois/<camera_name>/<roi_id>', methods=['DELETE'])
def delete_saved_roi(camera_name, roi_id):
    """Delete a saved ROI."""
    roi_dir = Path(SAVED_ROIS_PATH) / camera_name

    # Delete both JSON and image files
    json_path = roi_dir / f"{roi_id}.json"
    img_path = roi_dir / f"{roi_id}.png"

    deleted = False
    if json_path.exists():
        json_path.unlink()
        deleted = True
    if img_path.exists():
        img_path.unlink()
        deleted = True

    if deleted:
        return jsonify({'success': True})
    return jsonify({'error': 'ROI not found'}), 404


@app.route('/api/saved-rois/<camera_name>/<roi_id>/image')
def get_saved_roi_image(camera_name, roi_id):
    """Get saved ROI screenshot image."""
    img_path = Path(SAVED_ROIS_PATH) / camera_name / f"{roi_id}.png"
    if not img_path.exists():
        return jsonify({'error': 'Image not found'}), 404

    with open(img_path, 'rb') as f:
        return Response(f.read(), mimetype='image/png')


@app.route('/api/history-image/<camera_name>/<image_id>')
def get_history_image(camera_name, image_id):
    """Get history ROI image."""
    safe_name = re.sub(r'[^\w\-]', '_', camera_name)
    img_path = Path(HISTORY_IMAGES_PATH) / safe_name / f"{image_id}.jpg"
    if not img_path.exists():
        return jsonify({'error': 'Image not found'}), 404

    with open(img_path, 'rb') as f:
        return Response(f.read(), mimetype='image/jpeg')


@app.route('/api/discover', methods=['POST'])
def discover_cameras():
    """Discover cameras on the network using ONVIF, direct probing, and port scanning."""
    cameras = []
    seen_ips = set()

    # Try ONVIF WS-Discovery first
    logger.info("Starting ONVIF WS-Discovery...")
    onvif_cameras = ONVIFDiscovery.discover(timeout=5.0)
    for cam in onvif_cameras:
        if cam['ip'] not in seen_ips:
            cameras.append(cam)
            seen_ips.add(cam['ip'])
    logger.info(f"WS-Discovery found {len(onvif_cameras)} cameras")

    # Also try port scanning to find potential cameras
    logger.info("Starting port scan discovery...")
    port_cameras = PortScanner.scan_network(timeout=0.3)

    # Collect IPs that have camera-like ports open but weren't found by WS-Discovery
    ips_to_probe = []
    for cam in port_cameras:
        if cam['ip'] not in seen_ips:
            ips_to_probe.append(cam['ip'])
            # Add port-scanned cameras to the list
            cameras.append(cam)
            seen_ips.add(cam['ip'])

    # Try direct ONVIF probe on IPs found by port scanning
    if ips_to_probe:
        logger.info(f"Probing {len(ips_to_probe)} IPs for ONVIF devices...")
        direct_cameras = ONVIFDiscovery.probe_direct(ips_to_probe, timeout=2.0)

        # Update existing entries with ONVIF info if found
        for direct_cam in direct_cameras:
            # Find and update the existing entry
            for i, cam in enumerate(cameras):
                if cam['ip'] == direct_cam['ip']:
                    cameras[i] = direct_cam  # Replace with more detailed info
                    break

    logger.info(f"Total cameras discovered: {len(cameras)}")
    return jsonify(cameras)


@app.route('/api/ai-config', methods=['GET'])
def get_ai_config():
    """Get AI configuration."""
    return jsonify(AIService.get_config())


@app.route('/api/ai-config', methods=['POST'])
def save_ai_config():
    """Save AI configuration."""
    data = request.json
    config = AIProviderConfig(
        provider=data.get('provider', 'none'),
        api_key=data.get('api_key', ''),
        api_url=data.get('api_url', ''),
        model=data.get('model', ''),
        region=data.get('region', ''),
        enabled_for_ocr=data.get('enabled_for_ocr', False),
        enabled_for_description=data.get('enabled_for_description', False)
    )
    AIService.save_config(config)
    return jsonify({'success': True})


@app.route('/api/ai-test', methods=['POST'])
def test_ai():
    """Test AI connection with a sample image or camera."""
    data = request.json
    camera_name = data.get('camera')
    image_base64 = data.get('image')

    if not camera_name and not image_base64:
        return jsonify({'error': 'Camera name or image required'}), 400

    # Get image from camera if no uploaded image
    if not image_base64:
        camera_config = processor.cameras.get(camera_name)
        if not camera_config:
            return jsonify({'error': f'Camera not found: {camera_name}'}), 404

        # Capture frame
        frame, capture_error = processor.capture_frame(camera_config)
        if frame is None or capture_error:
            return jsonify({'error': capture_error or 'Failed to capture frame'}), 500

        # Extract ROI if defined
        if camera_config.roi_width > 0 and camera_config.roi_height > 0:
            x, y = camera_config.roi_x, camera_config.roi_y
            w, h = camera_config.roi_width, camera_config.roi_height
            img_h, img_w = frame.shape[:2]
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            roi_frame = frame[y:y+h, x:x+w]
            _, buffer = cv2.imencode('.jpg', roi_frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

    result = {'success': True}

    # Test scene description
    config = AIService.load_config()
    if config.enabled_for_description:
        try:
            description = AIService.describe_scene(image_base64)
            result['description'] = description
        except Exception as e:
            result['description_error'] = str(e)

    # Test OCR
    if config.enabled_for_ocr:
        try:
            ocr_result = AIService.enhance_ocr(image_base64)
            result['ocr'] = ocr_result
        except Exception as e:
            result['ocr_error'] = str(e)

    return jsonify(result)


@app.route('/api/ptz', methods=['POST'])
def ptz_control():
    """Control PTZ camera movement."""
    data = request.json
    camera_name = data.get('camera')
    direction = data.get('direction')

    if not camera_name or not direction:
        return jsonify({'error': 'Camera name and direction required'}), 400

    if direction not in ['up', 'down', 'left', 'right', 'home']:
        return jsonify({'error': 'Invalid direction. Use: up, down, left, right, home'}), 400

    camera_config = processor.cameras.get(camera_name)
    if not camera_config:
        return jsonify({'error': f'Camera not found: {camera_name}'}), 404

    result = PTZController.move(camera_config, direction)
    if result.get('error'):
        return jsonify(result), 500

    return jsonify(result)


@app.route('/api/saved-rois/<camera_name>/<roi_id>/validate', methods=['POST'])
def validate_roi(camera_name, roi_id):
    """Save user-validated value for a saved ROI for OCR training."""
    data = request.json
    validated_value = data.get('value')

    if validated_value is None:
        return jsonify({'error': 'Value required'}), 400

    # Load saved ROIs
    rois_file = os.path.join(SAVED_ROIS_PATH, f'{camera_name}.json')
    if not os.path.exists(rois_file):
        return jsonify({'error': 'No saved ROIs for this camera'}), 404

    try:
        with open(rois_file, 'r') as f:
            rois = json.load(f)

        # Find and update the ROI
        found = False
        for roi in rois:
            if roi.get('id') == roi_id:
                roi['validated_value'] = validated_value
                roi['validated_at'] = time.time()
                found = True
                break

        if not found:
            return jsonify({'error': 'ROI not found'}), 404

        # Save updated ROIs
        with open(rois_file, 'w') as f:
            json.dump(rois, f, indent=2)

        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error validating ROI: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/saved-rois/<camera_name>/train', methods=['POST'])
def train_ocr(camera_name):
    """Run OCR training to find optimal preprocessing for validated ROIs."""
    # Load saved ROIs with validated values
    rois_file = os.path.join(SAVED_ROIS_PATH, f'{camera_name}.json')
    if not os.path.exists(rois_file):
        return jsonify({'error': 'No saved ROIs for this camera'}), 404

    try:
        with open(rois_file, 'r') as f:
            rois = json.load(f)

        # Filter only validated ROIs
        validated_rois = [r for r in rois if r.get('validated_value') is not None]

        if len(validated_rois) == 0:
            return jsonify({'error': 'No validated ROIs found. Please validate at least one ROI first.'}), 400

        # Test different preprocessing options on validated ROIs
        preprocessing_options = ['auto', 'threshold', 'adaptive', 'invert', 'none']
        psm_modes = [7, 8, 6, 13, 11]  # Single line, word, block, raw, sparse

        results = []
        best_config = {'preprocessing': 'auto', 'psm': 7, 'accuracy': 0}

        for roi_data in validated_rois:
            roi_id = roi_data.get('id')
            expected_value = str(roi_data.get('validated_value'))

            # Load the ROI image
            img_path = os.path.join(SAVED_ROIS_PATH, camera_name, f'{roi_id}.png')
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            roi_results = {
                'roi_id': roi_id,
                'expected': expected_value,
                'tests': []
            }

            for preproc in preprocessing_options:
                for psm in psm_modes:
                    try:
                        # Process the image
                        processed = processor._preprocess_image(img, preproc)

                        # Run OCR with specific PSM mode
                        config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789.-'
                        text = pytesseract.image_to_string(processed, config=config).strip()

                        # Extract numeric value
                        numbers = re.findall(r'[-+]?\d*\.?\d+', text)
                        ocr_value = numbers[0] if numbers else ''

                        # Check if it matches expected
                        match = ocr_value == expected_value

                        roi_results['tests'].append({
                            'preprocessing': preproc,
                            'psm': psm,
                            'result': ocr_value,
                            'match': match
                        })

                        if match:
                            # Count matches for this config across all validated ROIs
                            config_key = f'{preproc}_{psm}'
                            current_count = sum(1 for r in results for t in r.get('tests', [])
                                              if t.get('match') and f"{t['preprocessing']}_{t['psm']}" == config_key)
                            if current_count + 1 > best_config['accuracy']:
                                best_config = {'preprocessing': preproc, 'psm': psm, 'accuracy': current_count + 1}

                    except Exception as e:
                        pass

            results.append(roi_results)

        # Calculate accuracy for each config
        config_stats = {}
        for roi_result in results:
            for test in roi_result.get('tests', []):
                key = f"{test['preprocessing']}_psm{test['psm']}"
                if key not in config_stats:
                    config_stats[key] = {'matches': 0, 'total': 0, 'preprocessing': test['preprocessing'], 'psm': test['psm']}
                config_stats[key]['total'] += 1
                if test.get('match'):
                    config_stats[key]['matches'] += 1

        # Sort by accuracy
        ranked_configs = sorted(
            config_stats.values(),
            key=lambda x: x['matches'] / x['total'] if x['total'] > 0 else 0,
            reverse=True
        )

        # Get the best config
        if ranked_configs and ranked_configs[0]['matches'] > 0:
            best = ranked_configs[0]
            best_config = {
                'preprocessing': best['preprocessing'],
                'psm': best['psm'],
                'accuracy': best['matches'] / best['total'] * 100,
                'matches': best['matches'],
                'total': best['total']
            }

        return jsonify({
            'success': True,
            'validated_count': len(validated_rois),
            'best_config': best_config,
            'ranked_configs': ranked_configs[:5],  # Top 5 configs
            'detailed_results': results
        })

    except Exception as e:
        logger.error(f"Error training OCR: {e}")
        return jsonify({'error': str(e)}), 500


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
    logger.info("Starting Camera OCR Add-on")
    processor.start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
