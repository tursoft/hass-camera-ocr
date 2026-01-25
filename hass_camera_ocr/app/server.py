#!/usr/bin/env python3
"""Camera OCR Add-on Server with Full Admin Interface and Template Matching."""

VERSION = "1.2.20"

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
from flask import Flask, jsonify, request, render_template, Response
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
    # AI configuration per camera
    ai_provider: str = "none"  # none, openai, anthropic, google, ollama, custom, google-vision, azure-ocr, aws-textract
    ai_api_key: str = ""
    ai_api_url: str = ""  # For Ollama, custom endpoints
    ai_model: str = ""
    ai_region: str = ""  # AWS region for Textract
    ai_enabled_for_ocr: bool = False
    ai_enabled_for_description: bool = False


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
    def describe_scene(cls, image_base64: str, camera: CameraConfig = None) -> str:
        """Generate scene description using AI."""
        # Use camera-specific config if provided, otherwise fall back to global
        if camera and camera.ai_provider != 'none':
            provider = camera.ai_provider
            enabled = camera.ai_enabled_for_description
            api_key = camera.ai_api_key
            api_url = camera.ai_api_url
            model = camera.ai_model
            region = camera.ai_region
        else:
            config = cls.load_config()
            provider = config.provider
            enabled = config.enabled_for_description
            api_key = config.api_key
            api_url = config.api_url
            model = config.model
            region = config.region

        if not enabled or provider == 'none':
            return ""

        try:
            if provider == 'openai' or provider == 'custom':
                return cls._call_openai(image_base64, "scene", api_key, api_url, model)
            elif provider == 'anthropic':
                return cls._call_anthropic(image_base64, "scene", api_key, model)
            elif provider == 'google':
                return cls._call_google(image_base64, "scene", api_key, model)
            elif provider == 'ollama':
                return cls._call_ollama(image_base64, "scene", api_url, model)
        except Exception as e:
            logger.error(f"AI scene description error: {e}")
            return f"Error: {str(e)}"

        return ""

    @classmethod
    def enhance_ocr(cls, image_base64: str, roi_image_base64: str = None, camera: CameraConfig = None) -> Optional[str]:
        """Use AI to extract/verify OCR value."""
        # Use camera-specific config if provided, otherwise fall back to global
        if camera and camera.ai_provider != 'none':
            provider = camera.ai_provider
            enabled = camera.ai_enabled_for_ocr
            api_key = camera.ai_api_key
            api_url = camera.ai_api_url
            model = camera.ai_model
            region = camera.ai_region
        else:
            config = cls.load_config()
            provider = config.provider
            enabled = config.enabled_for_ocr
            api_key = config.api_key
            api_url = config.api_url
            model = config.model
            region = config.region

        if not enabled or provider == 'none':
            return None

        image_to_use = roi_image_base64 or image_base64

        try:
            if provider == 'openai' or provider == 'custom':
                return cls._call_openai(image_to_use, "ocr", api_key, api_url, model)
            elif provider == 'anthropic':
                return cls._call_anthropic(image_to_use, "ocr", api_key, model)
            elif provider == 'google':
                return cls._call_google(image_to_use, "ocr", api_key, model)
            elif provider == 'ollama':
                return cls._call_ollama(image_to_use, "ocr", api_url, model)
            elif provider == 'google-vision':
                return cls._call_google_vision(image_to_use, api_key)
            elif provider == 'google-docai':
                return cls._call_google_docai(image_to_use, api_key, api_url)
            elif provider == 'azure-ocr':
                return cls._call_azure_ocr(image_to_use, api_key, api_url)
            elif provider == 'aws-textract':
                return cls._call_aws_textract(image_to_use, api_key, region)
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
    def _call_openai(cls, image_base64: str, task: str, api_key: str = None, api_url: str = None, model: str = None) -> str:
        """Call OpenAI API or OpenAI-compatible custom endpoint."""
        import urllib.request

        if not api_key:
            config = cls.load_config()
            api_key = config.api_key
            api_url = api_url or config.api_url
            model = model or config.model

        model = model or cls.DEFAULT_MODELS['openai']

        # Use custom URL if provided, otherwise use OpenAI default
        if api_url:
            api_url = api_url.rstrip('/')
            if not api_url.endswith('/chat/completions'):
                api_url = f'{api_url}/chat/completions'
        else:
            api_url = 'https://api.openai.com/v1/chat/completions'

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
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
    def _call_anthropic(cls, image_base64: str, task: str, api_key: str = None, model: str = None) -> str:
        """Call Anthropic API."""
        import urllib.request

        if not api_key:
            config = cls.load_config()
            api_key = config.api_key
            model = model or config.model

        model = model or cls.DEFAULT_MODELS['anthropic']

        headers = {
            'Content-Type': 'application/json',
            'x-api-key': api_key,
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
    def _call_google(cls, image_base64: str, task: str, api_key: str = None, model: str = None) -> str:
        """Call Google Gemini API."""
        import urllib.request

        if not api_key:
            config = cls.load_config()
            api_key = config.api_key
            model = model or config.model

        model = model or cls.DEFAULT_MODELS['google']

        url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}'

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
    def _call_ollama(cls, image_base64: str, task: str, api_url: str = None, model: str = None) -> str:
        """Call Ollama API (local)."""
        import urllib.request

        if not api_url:
            config = cls.load_config()
            api_url = config.api_url
            model = model or config.model

        model = model or cls.DEFAULT_MODELS['ollama']
        api_url = api_url or 'http://localhost:11434'

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
    def _call_google_vision(cls, image_base64: str, api_key: str = None) -> str:
        """Call Google Cloud Vision API for OCR."""
        import urllib.request

        if not api_key:
            config = cls.load_config()
            api_key = config.api_key

        url = f'https://vision.googleapis.com/v1/images:annotate?key={api_key}'

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
    def _call_google_docai(cls, image_base64: str, api_key: str = None, api_url: str = None) -> str:
        """Call Google Document AI API for OCR."""
        import urllib.request

        if not api_key:
            config = cls.load_config()
            api_key = config.api_key
            api_url = api_url or config.api_url

        # Document AI requires project ID and processor ID in the URL
        # Format: api_url should be like: projects/PROJECT_ID/locations/LOCATION/processors/PROCESSOR_ID
        processor_path = api_url or ''
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
    def _call_azure_ocr(cls, image_base64: str, api_key: str = None, api_url: str = None) -> str:
        """Call Azure Computer Vision API for OCR."""
        import urllib.request

        if not api_key:
            config = cls.load_config()
            api_key = config.api_key
            api_url = api_url or config.api_url

        endpoint = api_url or 'https://westus.api.cognitive.microsoft.com'
        endpoint = endpoint.rstrip('/')

        # First, submit the read request
        url = f'{endpoint}/vision/v3.2/read/analyze'

        image_data = base64.b64decode(image_base64)

        req = urllib.request.Request(
            url,
            data=image_data,
            headers={
                'Content-Type': 'application/octet-stream',
                'Ocp-Apim-Subscription-Key': api_key
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
    def _call_aws_textract(cls, image_base64: str, api_key: str = None, region: str = None) -> str:
        """Call AWS Textract API for OCR."""
        import urllib.request
        import hashlib
        import hmac
        from datetime import datetime

        if not api_key:
            config = cls.load_config()
            api_key = config.api_key
            region = region or config.region

        region = region or 'us-east-1'

        # AWS Textract requires signature v4 authentication
        # For simplicity, we'll use boto3 if available, otherwise basic API
        try:
            import boto3
            client = boto3.client(
                'textract',
                region_name=region,
                aws_access_key_id=api_key.split(':')[0] if ':' in api_key else api_key,
                aws_secret_access_key=api_key.split(':')[1] if ':' in api_key else ''
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

        # Generate AI scene description if enabled (use camera-specific config or global)
        if camera.ai_enabled_for_description or (camera.ai_provider == 'none' and AIService.load_config().enabled_for_description):
            try:
                _, buffer = cv2.imencode('.jpg', frame)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                result.video_description = AIService.describe_scene(image_base64, camera)
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
# API Routes
# ============================================================================

@app.route('/')
def index():
    """Serve web UI."""
    return render_template('index.html', version=VERSION)


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


@app.route('/api/history', methods=['DELETE'])
def clear_all_history():
    """Clear all value history."""
    processor.history = {}
    processor._save_history()
    return jsonify({'status': 'ok', 'message': 'All history cleared'})


@app.route('/api/history/<camera_name>', methods=['DELETE'])
def clear_camera_history(camera_name):
    """Clear value history for a specific camera."""
    from urllib.parse import unquote
    camera_name = unquote(camera_name)
    if camera_name in processor.history:
        del processor.history[camera_name]
        processor._save_history()
        return jsonify({'status': 'ok', 'message': f'History cleared for {camera_name}'})
    return jsonify({'status': 'ok', 'message': 'No history found for camera'})


@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """Get camera configurations (with sensitive data masked)."""
    result = {}
    for name, cam in processor.cameras.items():
        cam_dict = asdict(cam)
        # Mask sensitive credentials
        if cam_dict.get('password'):
            cam_dict['password'] = '••••••••'
        if cam_dict.get('ai_api_key'):
            cam_dict['ai_api_key'] = '••••••••'
        result[name] = cam_dict
    return jsonify(result)


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
        # AI configuration per camera
        'ai_provider': data.get('ai_provider', 'none'),
        'ai_api_key': data.get('ai_api_key', ''),
        'ai_api_url': data.get('ai_api_url', ''),
        'ai_model': data.get('ai_model', ''),
        'ai_region': data.get('ai_region', ''),
        'ai_enabled_for_ocr': data.get('ai_enabled_for_ocr', False),
        'ai_enabled_for_description': data.get('ai_enabled_for_description', False),
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
                # AI configuration per camera
                'ai_provider': data.get('ai_provider', cam.ai_provider),
                'ai_api_key': data.get('ai_api_key', cam.ai_api_key),
                'ai_api_url': data.get('ai_api_url', cam.ai_api_url),
                'ai_model': data.get('ai_model', cam.ai_model),
                'ai_region': data.get('ai_region', cam.ai_region),
                'ai_enabled_for_ocr': data.get('ai_enabled_for_ocr', cam.ai_enabled_for_ocr),
                'ai_enabled_for_description': data.get('ai_enabled_for_description', cam.ai_enabled_for_description),
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

    # Use camera-specific config if camera was provided, otherwise global config
    camera_config = processor.cameras.get(camera_name) if camera_name else None

    # Test scene description
    if camera_config and camera_config.ai_enabled_for_description:
        try:
            description = AIService.describe_scene(image_base64, camera_config)
            result['description'] = description
        except Exception as e:
            result['description_error'] = str(e)
    elif not camera_config:
        config = AIService.load_config()
        if config.enabled_for_description:
            try:
                description = AIService.describe_scene(image_base64)
                result['description'] = description
            except Exception as e:
                result['description_error'] = str(e)

    # Test OCR
    if camera_config and camera_config.ai_enabled_for_ocr:
        try:
            ocr_result = AIService.enhance_ocr(image_base64, camera=camera_config)
            result['ocr'] = ocr_result
        except Exception as e:
            result['ocr_error'] = str(e)
    elif not camera_config:
        config = AIService.load_config()
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
    from urllib.parse import unquote
    camera_name = unquote(camera_name)
    roi_id = unquote(roi_id)

    data = request.json
    validated_value = data.get('value')

    logger.info(f"Validating ROI: camera={camera_name}, roi_id={roi_id}, value={validated_value}")

    if validated_value is None:
        return jsonify({'error': 'Value required'}), 400

    # Load the specific ROI JSON file
    roi_file = Path(SAVED_ROIS_PATH) / camera_name / f'{roi_id}.json'
    if not roi_file.exists():
        logger.warning(f"ROI file not found: {roi_file}")
        return jsonify({'error': f'ROI not found: {roi_id}'}), 404

    try:
        with open(roi_file, 'r') as f:
            roi_data = json.load(f)

        logger.info(f"Updating ROI {roi_id} with validated value: {validated_value}")

        # Update the ROI with validated value
        roi_data['validated_value'] = validated_value
        roi_data['validated_at'] = time.time()

        # Save updated ROI
        with open(roi_file, 'w') as f:
            json.dump(roi_data, f, indent=2)

        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error validating ROI: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/saved-rois/<camera_name>/train', methods=['POST'])
def train_ocr(camera_name):
    """Run OCR training to find optimal preprocessing for validated ROIs."""
    from urllib.parse import unquote
    camera_name = unquote(camera_name)

    # Load saved ROIs from individual files in camera directory
    roi_dir = Path(SAVED_ROIS_PATH) / camera_name
    logger.info(f"Training OCR for camera: {camera_name}, ROI dir: {roi_dir}")

    if not roi_dir.exists():
        logger.warning(f"ROI directory not found: {roi_dir}")
        return jsonify({'error': f'No saved ROIs found for camera "{camera_name}". Save some ROIs first.'}), 404

    try:
        # Load all ROIs from individual JSON files
        rois = []
        for json_file in roi_dir.glob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    roi_data = json.load(f)
                    roi_data['id'] = json_file.stem  # Add ID from filename
                    rois.append(roi_data)
            except Exception as e:
                logger.error(f"Error loading ROI {json_file}: {e}")

        logger.info(f"Found {len(rois)} ROIs, checking for validated ones...")

        if len(rois) == 0:
            return jsonify({'error': f'No saved ROIs found for camera "{camera_name}". Save some ROIs first.'}), 404

        # Filter only validated ROIs
        validated_rois = [r for r in rois if r.get('validated_value') is not None]

        if len(validated_rois) == 0:
            return jsonify({'error': f'No validated ROIs found ({len(rois)} ROIs exist). Click on a saved ROI and use "Validate" to enter the correct value first.'}), 400

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
