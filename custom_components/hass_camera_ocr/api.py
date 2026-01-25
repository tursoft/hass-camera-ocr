"""API endpoints for Camera Data Extractor web panel."""
from __future__ import annotations

import logging
import base64
from typing import Any

from aiohttp import web
import cv2

from homeassistant.components.http import HomeAssistantView
from homeassistant.core import HomeAssistant

from .const import DOMAIN
from .camera_processor import CameraProcessor
from .template_matcher import TemplateMatcher

_LOGGER = logging.getLogger(__name__)


class CameraDataExtractorAPIView(HomeAssistantView):
    """Base API view for Camera Data Extractor."""

    requires_auth = True

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the API view."""
        self.hass = hass

    def _get_processor(self, entry_id: str) -> CameraProcessor | None:
        """Get camera processor for an entry."""
        data = self.hass.data.get(DOMAIN, {}).get(entry_id, {})
        if isinstance(data, dict) and "processor" in data:
            return data["processor"]
        return None

    def _get_matcher(self) -> TemplateMatcher:
        """Get or create template matcher."""
        if "matcher" not in self.hass.data.get(DOMAIN, {}):
            self.hass.data[DOMAIN]["matcher"] = TemplateMatcher()
        return self.hass.data[DOMAIN]["matcher"]


class CaptureFrameView(CameraDataExtractorAPIView):
    """API endpoint to capture a frame from a camera."""

    url = "/api/hass_camera_ocr/capture/{entry_id}"
    name = "api:hass_camera_ocr:capture"

    async def get(self, request: web.Request, entry_id: str) -> web.Response:
        """Capture a frame and return as base64 PNG."""
        try:
            data = self.hass.data.get(DOMAIN, {}).get(entry_id, {})

            if not data:
                return web.json_response(
                    {"error": "Entry not found"}, status=404
                )

            # Get stream URL and credentials
            stream_url = data.get("stream_url")
            username = data.get("username")
            password = data.get("password")

            if not stream_url:
                return web.json_response(
                    {"error": "No stream URL configured"}, status=400
                )

            # Capture frame
            processor = CameraProcessor(stream_url, username, password)
            result = await self.hass.async_add_executor_job(processor.capture_frame)

            if not result.success:
                return web.json_response(
                    {"error": result.error or "Failed to capture frame"}, status=500
                )

            # Encode frame to base64 PNG
            _, buffer = cv2.imencode(".png", result.frame)
            frame_b64 = base64.b64encode(buffer).decode("utf-8")

            # Get frame dimensions
            h, w = result.frame.shape[:2]

            return web.json_response({
                "success": True,
                "frame": frame_b64,
                "width": w,
                "height": h,
                "timestamp": result.timestamp,
            })

        except Exception as ex:
            _LOGGER.error("Error capturing frame: %s", ex)
            return web.json_response(
                {"error": str(ex)}, status=500
            )


class SaveTemplateView(CameraDataExtractorAPIView):
    """API endpoint to save a reference template."""

    url = "/api/hass_camera_ocr/template/save"
    name = "api:hass_camera_ocr:template:save"

    async def post(self, request: web.Request) -> web.Response:
        """Save a reference template from the current frame."""
        try:
            body = await request.json()

            entry_id = body.get("entry_id")
            template_name = body.get("name")
            roi = body.get("roi")  # {x, y, width, height}

            if not all([entry_id, template_name, roi]):
                return web.json_response(
                    {"error": "Missing required fields"}, status=400
                )

            data = self.hass.data.get(DOMAIN, {}).get(entry_id, {})
            if not data:
                return web.json_response(
                    {"error": "Entry not found"}, status=404
                )

            # Capture current frame
            stream_url = data.get("stream_url")
            username = data.get("username")
            password = data.get("password")

            processor = CameraProcessor(stream_url, username, password)
            result = await self.hass.async_add_executor_job(processor.capture_frame)

            if not result.success:
                return web.json_response(
                    {"error": "Failed to capture frame"}, status=500
                )

            # Save template
            matcher = self._get_matcher()
            roi_tuple = (
                int(roi["x"]),
                int(roi["y"]),
                int(roi["width"]),
                int(roi["height"]),
            )

            success = await self.hass.async_add_executor_job(
                matcher.save_reference,
                template_name,
                result.frame,
                roi_tuple,
            )

            if success:
                # Store template name in entry data
                if "templates" not in data:
                    data["templates"] = []
                if template_name not in data["templates"]:
                    data["templates"].append(template_name)

                return web.json_response({
                    "success": True,
                    "message": f"Template '{template_name}' saved successfully",
                })
            else:
                return web.json_response(
                    {"error": "Failed to save template"}, status=500
                )

        except Exception as ex:
            _LOGGER.error("Error saving template: %s", ex)
            return web.json_response(
                {"error": str(ex)}, status=500
            )


class ListTemplatesView(CameraDataExtractorAPIView):
    """API endpoint to list available templates."""

    url = "/api/hass_camera_ocr/templates"
    name = "api:hass_camera_ocr:templates:list"

    async def get(self, request: web.Request) -> web.Response:
        """List all available templates."""
        try:
            matcher = self._get_matcher()
            templates = await self.hass.async_add_executor_job(matcher.list_templates)

            return web.json_response({
                "success": True,
                "templates": templates,
            })

        except Exception as ex:
            _LOGGER.error("Error listing templates: %s", ex)
            return web.json_response(
                {"error": str(ex)}, status=500
            )


class TemplatePreviewView(CameraDataExtractorAPIView):
    """API endpoint to get template preview."""

    url = "/api/hass_camera_ocr/template/preview/{name}"
    name = "api:hass_camera_ocr:template:preview"

    async def get(self, request: web.Request, name: str) -> web.Response:
        """Get a preview image of a template."""
        try:
            matcher = self._get_matcher()
            preview = await self.hass.async_add_executor_job(
                matcher.get_template_preview, name
            )

            if preview:
                return web.json_response({
                    "success": True,
                    "preview": preview,
                })
            else:
                return web.json_response(
                    {"error": "Template not found"}, status=404
                )

        except Exception as ex:
            _LOGGER.error("Error getting template preview: %s", ex)
            return web.json_response(
                {"error": str(ex)}, status=500
            )


class FindRegionView(CameraDataExtractorAPIView):
    """API endpoint to find region in current frame using template."""

    url = "/api/hass_camera_ocr/find"
    name = "api:hass_camera_ocr:find"

    async def post(self, request: web.Request) -> web.Response:
        """Find the template region in the current frame."""
        try:
            body = await request.json()

            entry_id = body.get("entry_id")
            template_name = body.get("template_name")

            if not all([entry_id, template_name]):
                return web.json_response(
                    {"error": "Missing required fields"}, status=400
                )

            data = self.hass.data.get(DOMAIN, {}).get(entry_id, {})
            if not data:
                return web.json_response(
                    {"error": "Entry not found"}, status=404
                )

            # Capture current frame
            stream_url = data.get("stream_url")
            username = data.get("username")
            password = data.get("password")

            processor = CameraProcessor(stream_url, username, password)
            frame_result = await self.hass.async_add_executor_job(
                processor.capture_frame
            )

            if not frame_result.success:
                return web.json_response(
                    {"error": "Failed to capture frame"}, status=500
                )

            # Find region
            matcher = self._get_matcher()
            match_result = await self.hass.async_add_executor_job(
                matcher.find_region,
                frame_result.frame,
                template_name,
            )

            # Encode frame with match visualization
            if match_result.found:
                vis_frame = frame_result.frame.copy()
                cv2.rectangle(
                    vis_frame,
                    (match_result.x, match_result.y),
                    (match_result.x + match_result.width, match_result.y + match_result.height),
                    (0, 255, 0),
                    2,
                )
                _, buffer = cv2.imencode(".png", vis_frame)
                frame_b64 = base64.b64encode(buffer).decode("utf-8")
            else:
                _, buffer = cv2.imencode(".png", frame_result.frame)
                frame_b64 = base64.b64encode(buffer).decode("utf-8")

            return web.json_response({
                "success": True,
                "found": match_result.found,
                "roi": {
                    "x": match_result.x,
                    "y": match_result.y,
                    "width": match_result.width,
                    "height": match_result.height,
                },
                "confidence": match_result.confidence,
                "angle": match_result.angle,
                "frame": frame_b64,
                "error": match_result.error,
            })

        except Exception as ex:
            _LOGGER.error("Error finding region: %s", ex)
            return web.json_response(
                {"error": str(ex)}, status=500
            )


class DeleteTemplateView(CameraDataExtractorAPIView):
    """API endpoint to delete a template."""

    url = "/api/hass_camera_ocr/template/delete/{name}"
    name = "api:hass_camera_ocr:template:delete"

    async def delete(self, request: web.Request, name: str) -> web.Response:
        """Delete a template."""
        try:
            matcher = self._get_matcher()
            success = await self.hass.async_add_executor_job(
                matcher.delete_template, name
            )

            if success:
                return web.json_response({
                    "success": True,
                    "message": f"Template '{name}' deleted",
                })
            else:
                return web.json_response(
                    {"error": "Failed to delete template"}, status=500
                )

        except Exception as ex:
            _LOGGER.error("Error deleting template: %s", ex)
            return web.json_response(
                {"error": str(ex)}, status=500
            )


class ListEntriesView(CameraDataExtractorAPIView):
    """API endpoint to list all configured camera entries."""

    url = "/api/hass_camera_ocr/entries"
    name = "api:hass_camera_ocr:entries"

    async def get(self, request: web.Request) -> web.Response:
        """List all configured camera entries."""
        try:
            entries = []
            config_entries = self.hass.config_entries.async_entries(DOMAIN)

            for entry in config_entries:
                entries.append({
                    "entry_id": entry.entry_id,
                    "title": entry.title,
                    "camera_name": entry.data.get("camera_name"),
                    "value_name": entry.data.get("value_name"),
                    "stream_url": entry.data.get("stream_url"),
                })

            return web.json_response({
                "success": True,
                "entries": entries,
            })

        except Exception as ex:
            _LOGGER.error("Error listing entries: %s", ex)
            return web.json_response(
                {"error": str(ex)}, status=500
            )


def async_register_api(hass: HomeAssistant) -> None:
    """Register API endpoints."""
    hass.http.register_view(CaptureFrameView(hass))
    hass.http.register_view(SaveTemplateView(hass))
    hass.http.register_view(ListTemplatesView(hass))
    hass.http.register_view(TemplatePreviewView(hass))
    hass.http.register_view(FindRegionView(hass))
    hass.http.register_view(DeleteTemplateView(hass))
    hass.http.register_view(ListEntriesView(hass))
