"""Template matching for finding regions in rotated/moved camera views."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np

_LOGGER = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result from template matching."""

    found: bool
    x: int
    y: int
    width: int
    height: int
    confidence: float
    scale: float = 1.0
    angle: float = 0.0
    error: str | None = None


@dataclass
class ReferenceTemplate:
    """Reference template for matching."""

    name: str
    image: np.ndarray
    roi: tuple[int, int, int, int]  # x, y, width, height
    features: Any = None  # ORB features for feature-based matching


class TemplateMatcher:
    """Matcher for finding reference regions in frames with rotation/scale support."""

    def __init__(self, storage_path: str | None = None) -> None:
        """Initialize the template matcher.

        Args:
            storage_path: Path to store reference templates
        """
        self._storage_path = storage_path or "/config/custom_components/hass_camera_ocr/templates"
        self._templates: dict[str, ReferenceTemplate] = {}
        self._orb = cv2.ORB_create(nfeatures=500)
        self._bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Ensure storage directory exists
        Path(self._storage_path).mkdir(parents=True, exist_ok=True)

    def save_reference(
        self,
        name: str,
        frame: np.ndarray,
        roi: tuple[int, int, int, int],
    ) -> bool:
        """Save a reference template from a frame.

        Args:
            name: Unique name for this reference
            frame: Full frame image
            roi: Region of interest (x, y, width, height)

        Returns:
            True if saved successfully
        """
        try:
            x, y, w, h = roi

            # Extract the ROI region for template matching context
            # Use a larger area around ROI for better matching
            padding = 50
            img_h, img_w = frame.shape[:2]

            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img_w, x + w + padding)
            y2 = min(img_h, y + h + padding)

            template_region = frame[y1:y2, x1:x2].copy()

            # Calculate relative ROI within the template region
            relative_roi = (
                x - x1,
                y - y1,
                w,
                h,
            )

            # Extract ORB features
            gray = cv2.cvtColor(template_region, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self._orb.detectAndCompute(gray, None)

            # Create template object
            template = ReferenceTemplate(
                name=name,
                image=template_region,
                roi=relative_roi,
                features=(keypoints, descriptors) if descriptors is not None else None,
            )

            self._templates[name] = template

            # Save to disk
            self._save_template_to_disk(name, template, frame, roi)

            _LOGGER.info("Saved reference template '%s' with ROI %s", name, roi)
            return True

        except Exception as ex:
            _LOGGER.error("Failed to save reference template: %s", ex)
            return False

    def _save_template_to_disk(
        self,
        name: str,
        template: ReferenceTemplate,
        full_frame: np.ndarray,
        original_roi: tuple[int, int, int, int],
    ) -> None:
        """Save template data to disk."""
        safe_name = name.replace(" ", "_").replace("/", "_")
        base_path = Path(self._storage_path) / safe_name

        # Save template image
        cv2.imwrite(str(base_path) + "_template.png", template.image)

        # Save full frame for reference
        cv2.imwrite(str(base_path) + "_fullframe.png", full_frame)

        # Save metadata
        metadata = {
            "name": name,
            "original_roi": original_roi,
            "relative_roi": template.roi,
            "template_shape": template.image.shape[:2],
        }
        with open(str(base_path) + "_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load_template(self, name: str) -> bool:
        """Load a template from disk.

        Args:
            name: Template name

        Returns:
            True if loaded successfully
        """
        try:
            safe_name = name.replace(" ", "_").replace("/", "_")
            base_path = Path(self._storage_path) / safe_name

            template_path = str(base_path) + "_template.png"
            metadata_path = str(base_path) + "_metadata.json"

            if not os.path.exists(template_path) or not os.path.exists(metadata_path):
                return False

            # Load template image
            template_image = cv2.imread(template_path)
            if template_image is None:
                return False

            # Load metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Extract features
            gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self._orb.detectAndCompute(gray, None)

            # Create template object
            template = ReferenceTemplate(
                name=name,
                image=template_image,
                roi=tuple(metadata["relative_roi"]),
                features=(keypoints, descriptors) if descriptors is not None else None,
            )

            self._templates[name] = template
            _LOGGER.info("Loaded template '%s'", name)
            return True

        except Exception as ex:
            _LOGGER.error("Failed to load template '%s': %s", name, ex)
            return False

    def find_region(
        self,
        frame: np.ndarray,
        template_name: str,
        use_feature_matching: bool = True,
    ) -> MatchResult:
        """Find the ROI region in a frame using the saved template.

        Args:
            frame: Current frame to search in
            template_name: Name of the reference template
            use_feature_matching: Use feature-based matching (better for rotation)

        Returns:
            MatchResult with the found region
        """
        if template_name not in self._templates:
            # Try to load from disk
            if not self.load_template(template_name):
                return MatchResult(
                    found=False,
                    x=0, y=0, width=0, height=0,
                    confidence=0,
                    error=f"Template '{template_name}' not found",
                )

        template = self._templates[template_name]

        if use_feature_matching and template.features is not None:
            return self._feature_match(frame, template)
        else:
            return self._template_match(frame, template)

    def _template_match(
        self,
        frame: np.ndarray,
        template: ReferenceTemplate,
    ) -> MatchResult:
        """Use template matching to find the region."""
        try:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template.image, cv2.COLOR_BGR2GRAY)

            # Multi-scale template matching
            best_match = None
            best_confidence = 0

            for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
                scaled_template = cv2.resize(
                    template_gray,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
                )

                if (scaled_template.shape[0] > frame_gray.shape[0] or
                    scaled_template.shape[1] > frame_gray.shape[1]):
                    continue

                result = cv2.matchTemplate(
                    frame_gray, scaled_template, cv2.TM_CCOEFF_NORMED
                )
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val > best_confidence:
                    best_confidence = max_val
                    th, tw = scaled_template.shape[:2]
                    # Calculate ROI position in the matched region
                    rx, ry, rw, rh = template.roi
                    best_match = MatchResult(
                        found=True,
                        x=int(max_loc[0] + rx * scale),
                        y=int(max_loc[1] + ry * scale),
                        width=int(rw * scale),
                        height=int(rh * scale),
                        confidence=max_val * 100,
                        scale=scale,
                    )

            if best_match and best_confidence > 0.5:
                return best_match

            return MatchResult(
                found=False,
                x=0, y=0, width=0, height=0,
                confidence=best_confidence * 100,
                error="Template match confidence too low",
            )

        except Exception as ex:
            _LOGGER.error("Template matching failed: %s", ex)
            return MatchResult(
                found=False,
                x=0, y=0, width=0, height=0,
                confidence=0,
                error=str(ex),
            )

    def _feature_match(
        self,
        frame: np.ndarray,
        template: ReferenceTemplate,
    ) -> MatchResult:
        """Use feature matching to find the region (handles rotation better)."""
        try:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template.image, cv2.COLOR_BGR2GRAY)

            # Get template features
            template_kp, template_desc = template.features

            if template_desc is None or len(template_kp) < 4:
                # Fall back to template matching
                return self._template_match(frame, template)

            # Extract features from frame
            frame_kp, frame_desc = self._orb.detectAndCompute(frame_gray, None)

            if frame_desc is None or len(frame_kp) < 4:
                return self._template_match(frame, template)

            # Match features
            matches = self._bf_matcher.match(template_desc, frame_desc)
            matches = sorted(matches, key=lambda x: x.distance)

            # Need at least 4 good matches for homography
            good_matches = [m for m in matches if m.distance < 50]

            if len(good_matches) < 4:
                # Fall back to template matching
                return self._template_match(frame, template)

            # Extract matched keypoints
            src_pts = np.float32(
                [template_kp[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [frame_kp[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)

            # Find homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is None:
                return self._template_match(frame, template)

            # Transform the ROI corners
            rx, ry, rw, rh = template.roi
            roi_corners = np.float32([
                [rx, ry],
                [rx + rw, ry],
                [rx + rw, ry + rh],
                [rx, ry + rh],
            ]).reshape(-1, 1, 2)

            transformed_corners = cv2.perspectiveTransform(roi_corners, M)

            # Get bounding box of transformed ROI
            x_coords = transformed_corners[:, 0, 0]
            y_coords = transformed_corners[:, 0, 1]

            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            # Calculate confidence based on number of inliers
            inliers = mask.sum() if mask is not None else 0
            confidence = (inliers / len(good_matches)) * 100

            # Calculate rotation angle from homography
            angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))

            return MatchResult(
                found=True,
                x=x_min,
                y=y_min,
                width=x_max - x_min,
                height=y_max - y_min,
                confidence=confidence,
                angle=angle,
            )

        except Exception as ex:
            _LOGGER.error("Feature matching failed: %s", ex)
            return self._template_match(frame, template)

    def get_template_preview(self, name: str) -> str | None:
        """Get a base64-encoded preview of a template.

        Args:
            name: Template name

        Returns:
            Base64-encoded PNG image or None
        """
        if name not in self._templates:
            if not self.load_template(name):
                return None

        template = self._templates[name]

        # Draw ROI rectangle on template
        preview = template.image.copy()
        x, y, w, h = template.roi
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Encode to base64
        _, buffer = cv2.imencode(".png", preview)
        return base64.b64encode(buffer).decode("utf-8")

    def list_templates(self) -> list[str]:
        """List all available templates.

        Returns:
            List of template names
        """
        templates = list(self._templates.keys())

        # Also check disk for templates not in memory
        try:
            for f in Path(self._storage_path).glob("*_metadata.json"):
                with open(f, "r") as fp:
                    metadata = json.load(fp)
                    name = metadata.get("name")
                    if name and name not in templates:
                        templates.append(name)
        except Exception:
            pass

        return templates

    def delete_template(self, name: str) -> bool:
        """Delete a template.

        Args:
            name: Template name

        Returns:
            True if deleted successfully
        """
        try:
            # Remove from memory
            if name in self._templates:
                del self._templates[name]

            # Remove from disk
            safe_name = name.replace(" ", "_").replace("/", "_")
            base_path = Path(self._storage_path) / safe_name

            for suffix in ["_template.png", "_fullframe.png", "_metadata.json"]:
                path = Path(str(base_path) + suffix)
                if path.exists():
                    path.unlink()

            return True
        except Exception as ex:
            _LOGGER.error("Failed to delete template '%s': %s", name, ex)
            return False
