"""Camera discovery using ONVIF and network scanning."""
from __future__ import annotations

import asyncio
import logging
import socket
from dataclasses import dataclass
from typing import Any
import xml.etree.ElementTree as ET

_LOGGER = logging.getLogger(__name__)

# ONVIF WS-Discovery message
ONVIF_PROBE = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"
    xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing"
    xmlns:wsd="http://schemas.xmlsoap.org/ws/2005/04/discovery"
    xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
    <soap:Header>
        <wsa:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</wsa:Action>
        <wsa:MessageID>uuid:{message_id}</wsa:MessageID>
        <wsa:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</wsa:To>
    </soap:Header>
    <soap:Body>
        <wsd:Probe>
            <wsd:Types>dn:NetworkVideoTransmitter</wsd:Types>
        </wsd:Probe>
    </soap:Body>
</soap:Envelope>"""


@dataclass
class DiscoveredCamera:
    """Discovered camera information."""

    ip: str
    port: int
    name: str
    manufacturer: str | None = None
    model: str | None = None
    xaddrs: str | None = None
    streams: list[str] | None = None


class CameraDiscovery:
    """Discover cameras on the network using ONVIF WS-Discovery."""

    MULTICAST_IP = "239.255.255.250"
    MULTICAST_PORT = 3702
    TIMEOUT = 5

    def __init__(self) -> None:
        """Initialize the camera discovery."""
        self._discovered: dict[str, DiscoveredCamera] = {}

    async def discover(self, timeout: float = 5.0) -> list[DiscoveredCamera]:
        """Discover cameras on the network.

        Args:
            timeout: Discovery timeout in seconds

        Returns:
            List of discovered cameras
        """
        self._discovered.clear()

        # Run ONVIF discovery
        await self._onvif_discover(timeout)

        # Also do a quick port scan for common camera ports
        await self._scan_common_ports()

        return list(self._discovered.values())

    async def _onvif_discover(self, timeout: float) -> None:
        """Discover ONVIF cameras using WS-Discovery."""
        try:
            import uuid

            # Create discovery message
            message_id = str(uuid.uuid4())
            probe_message = ONVIF_PROBE.format(message_id=message_id).encode("utf-8")

            # Create UDP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            sock.settimeout(0.5)

            # Send probe
            sock.sendto(probe_message, (self.MULTICAST_IP, self.MULTICAST_PORT))

            # Collect responses
            start_time = asyncio.get_event_loop().time()

            while (asyncio.get_event_loop().time() - start_time) < timeout:
                try:
                    data, addr = sock.recvfrom(65535)
                    await self._parse_onvif_response(data, addr[0])
                except socket.timeout:
                    continue
                except Exception as ex:
                    _LOGGER.debug("Error receiving discovery response: %s", ex)

            sock.close()

        except Exception as ex:
            _LOGGER.error("ONVIF discovery error: %s", ex)

    async def _parse_onvif_response(self, data: bytes, ip: str) -> None:
        """Parse ONVIF WS-Discovery response."""
        try:
            # Parse XML response
            root = ET.fromstring(data.decode("utf-8"))

            # Define namespaces
            ns = {
                "soap": "http://www.w3.org/2003/05/soap-envelope",
                "wsd": "http://schemas.xmlsoap.org/ws/2005/04/discovery",
                "wsa": "http://schemas.xmlsoap.org/ws/2004/08/addressing",
            }

            # Find XAddrs (service URLs)
            xaddrs_elem = root.find(".//wsd:XAddrs", ns)
            xaddrs = xaddrs_elem.text if xaddrs_elem is not None else None

            # Find scopes to get device info
            scopes_elem = root.find(".//wsd:Scopes", ns)
            scopes = scopes_elem.text.split() if scopes_elem is not None else []

            # Parse scopes for device info
            name = f"Camera at {ip}"
            manufacturer = None
            model = None

            for scope in scopes:
                if "name/" in scope.lower():
                    name = scope.split("/")[-1]
                elif "hardware/" in scope.lower() or "model/" in scope.lower():
                    model = scope.split("/")[-1]
                elif "manufacturer/" in scope.lower() or "mfg/" in scope.lower():
                    manufacturer = scope.split("/")[-1]

            # Extract port from XAddrs
            port = 80
            if xaddrs:
                for addr in xaddrs.split():
                    if "://" in addr:
                        try:
                            from urllib.parse import urlparse
                            parsed = urlparse(addr)
                            if parsed.port:
                                port = parsed.port
                            break
                        except Exception:
                            pass

            # Create discovered camera
            camera = DiscoveredCamera(
                ip=ip,
                port=port,
                name=name,
                manufacturer=manufacturer,
                model=model,
                xaddrs=xaddrs,
            )

            # Try to get streams
            streams = await self._get_common_streams(ip, port)
            camera.streams = streams

            self._discovered[ip] = camera
            _LOGGER.info("Discovered camera: %s (%s)", name, ip)

        except Exception as ex:
            _LOGGER.debug("Error parsing ONVIF response from %s: %s", ip, ex)

    async def _scan_common_ports(self) -> None:
        """Scan for cameras on common RTSP/HTTP ports."""
        # Get local network range
        try:
            local_ip = self._get_local_ip()
            if not local_ip:
                return

            # Get network prefix (assuming /24)
            prefix = ".".join(local_ip.split(".")[:-1])

            # Scan common camera ports
            common_ports = [554, 8554, 80, 8080, 8000]

            # Only scan a subset to avoid timeout
            tasks = []
            for i in range(1, 255):
                ip = f"{prefix}.{i}"
                if ip not in self._discovered:
                    for port in common_ports[:2]:  # Only check RTSP ports
                        tasks.append(self._check_camera_port(ip, port))

            # Run with concurrency limit
            sem = asyncio.Semaphore(50)

            async def limited_check(coro):
                async with sem:
                    return await coro

            await asyncio.gather(
                *[limited_check(task) for task in tasks],
                return_exceptions=True,
            )

        except Exception as ex:
            _LOGGER.debug("Error in port scan: %s", ex)

    async def _check_camera_port(self, ip: str, port: int) -> None:
        """Check if a camera is responding on the given port."""
        try:
            # Quick TCP connect check
            future = asyncio.open_connection(ip, port)
            reader, writer = await asyncio.wait_for(future, timeout=0.5)
            writer.close()
            await writer.wait_closed()

            # Port is open, might be a camera
            if ip not in self._discovered:
                streams = await self._get_common_streams(ip, port)
                if streams:
                    self._discovered[ip] = DiscoveredCamera(
                        ip=ip,
                        port=port,
                        name=f"Camera at {ip}",
                        streams=streams,
                    )

        except (asyncio.TimeoutError, OSError, ConnectionRefusedError):
            pass
        except Exception:
            pass

    async def _get_common_streams(self, ip: str, port: int = 554) -> list[str]:
        """Get common RTSP stream URLs to try."""
        streams = []

        # Common RTSP paths for various camera brands
        common_paths = [
            "/",
            "/stream1",
            "/stream0",
            "/live",
            "/live/ch00_0",
            "/cam/realmonitor",
            "/h264_ulaw.sdp",
            "/video1",
            "/video.mp4",
            "/media/video1",
            "/Streaming/Channels/101",
            "/Streaming/Channels/1",
            "/ch0_0.h264",
            "/0/1:1/main",
            "/videoMain",
            "/cam1/h264",
        ]

        for path in common_paths:
            if port == 554 or port == 8554:
                streams.append(f"rtsp://{ip}:{port}{path}")
            else:
                streams.append(f"http://{ip}:{port}{path}")

        return streams

    def _get_local_ip(self) -> str | None:
        """Get the local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return None

    async def get_camera_streams(
        self,
        ip: str,
        port: int = 554,
        username: str | None = None,
        password: str | None = None,
    ) -> list[str]:
        """Try to get working stream URLs from a camera.

        Args:
            ip: Camera IP address
            port: Camera port
            username: Optional username
            password: Optional password

        Returns:
            List of working stream URLs
        """
        import cv2

        working_streams = []
        streams = await self._get_common_streams(ip, port)

        for stream_url in streams[:5]:  # Only test first 5 to avoid long waits
            # Add credentials if provided
            if username and password:
                if "://" in stream_url:
                    protocol, rest = stream_url.split("://", 1)
                    test_url = f"{protocol}://{username}:{password}@{rest}"
                else:
                    test_url = stream_url
            else:
                test_url = stream_url

            # Test connection
            try:
                def test_stream():
                    cap = cv2.VideoCapture(test_url)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if cap.isOpened():
                        ret, _ = cap.read()
                        cap.release()
                        return ret
                    return False

                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, test_stream),
                    timeout=5.0,
                )

                if result:
                    working_streams.append(stream_url)
                    _LOGGER.info("Found working stream: %s", stream_url)

            except Exception:
                continue

        return working_streams


async def discover_cameras(timeout: float = 5.0) -> list[DiscoveredCamera]:
    """Convenience function to discover cameras.

    Args:
        timeout: Discovery timeout in seconds

    Returns:
        List of discovered cameras
    """
    discovery = CameraDiscovery()
    return await discovery.discover(timeout)
