"""OCR Engine for extracting numeric values from images."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pytesseract

_LOGGER = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result from OCR extraction."""

    value: float | None
    raw_text: str
    confidence: float
    success: bool
    error: str | None = None


class OCREngine:
    """Engine for extracting numeric values from images using OCR."""

    # Tesseract config for numeric extraction
    TESSERACT_CONFIG = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.-"

    def __init__(self, decimal_places: int = 1) -> None:
        """Initialize the OCR engine."""
        self.decimal_places = decimal_places

    def extract_value(
        self,
        image: np.ndarray,
        preprocessing: str = "auto",
        roi: tuple[int, int, int, int] | None = None,
    ) -> OCRResult:
        """Extract a numeric value from an image.

        Args:
            image: OpenCV image (BGR format)
            preprocessing: Preprocessing method to use
            roi: Region of interest (x, y, width, height), None for full image

        Returns:
            OCRResult with extracted value and metadata
        """
        try:
            # Crop to ROI if specified
            if roi and roi[2] > 0 and roi[3] > 0:
                x, y, w, h = roi
                # Ensure ROI is within image bounds
                img_h, img_w = image.shape[:2]
                x = max(0, min(x, img_w - 1))
                y = max(0, min(y, img_h - 1))
                w = min(w, img_w - x)
                h = min(h, img_h - y)
                image = image[y : y + h, x : x + w]

            # Preprocess the image
            processed = self._preprocess(image, preprocessing)

            # Run OCR
            raw_text = pytesseract.image_to_string(
                processed, config=self.TESSERACT_CONFIG
            ).strip()

            # Get confidence
            data = pytesseract.image_to_data(
                processed, config=self.TESSERACT_CONFIG, output_type=pytesseract.Output.DICT
            )
            confidences = [
                int(c) for c in data["conf"] if isinstance(c, (int, str)) and str(c).isdigit() and int(c) > 0
            ]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            # Parse numeric value
            value = self._parse_numeric_value(raw_text)

            if value is not None:
                return OCRResult(
                    value=round(value, self.decimal_places),
                    raw_text=raw_text,
                    confidence=avg_confidence,
                    success=True,
                )
            else:
                return OCRResult(
                    value=None,
                    raw_text=raw_text,
                    confidence=avg_confidence,
                    success=False,
                    error="Could not parse numeric value from text",
                )

        except Exception as ex:
            _LOGGER.error("OCR extraction failed: %s", ex)
            return OCRResult(
                value=None,
                raw_text="",
                confidence=0,
                success=False,
                error=str(ex),
            )

    def _preprocess(self, image: np.ndarray, method: str) -> np.ndarray:
        """Preprocess image for better OCR results.

        Args:
            image: OpenCV image (BGR format)
            method: Preprocessing method

        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Scale up small images for better OCR
        h, w = gray.shape[:2]
        if h < 100 or w < 100:
            scale = max(100 / h, 100 / w, 2)
            gray = cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )

        if method == "none":
            return gray

        if method == "threshold":
            # Simple binary threshold
            _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return processed

        if method == "adaptive":
            # Adaptive threshold for varying lighting
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            return processed

        if method == "invert":
            # Invert colors (for light text on dark background)
            _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return cv2.bitwise_not(processed)

        # Auto mode - try multiple approaches and pick best
        if method == "auto":
            return self._auto_preprocess(gray)

        return gray

    def _auto_preprocess(self, gray: np.ndarray) -> np.ndarray:
        """Automatically determine best preprocessing.

        Args:
            gray: Grayscale image

        Returns:
            Best preprocessed image
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Try to determine if it's light text on dark or vice versa
        mean_brightness = np.mean(gray)

        if mean_brightness < 127:
            # Dark background - likely need to invert
            _, binary = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            # Check if we need to invert
            if np.mean(binary) > 127:
                return binary
            return cv2.bitwise_not(binary)
        else:
            # Light background
            _, binary = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return binary

    def _parse_numeric_value(self, text: str) -> float | None:
        """Parse a numeric value from OCR text.

        Args:
            text: Raw OCR text

        Returns:
            Parsed float value or None if parsing fails
        """
        if not text:
            return None

        # Clean up the text
        text = text.strip()

        # Try to find a number pattern
        # Matches integers, decimals with . or ,
        patterns = [
            r"-?\d+\.\d+",  # Decimal with dot
            r"-?\d+,\d+",  # Decimal with comma
            r"-?\d+",  # Integer
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value_str = match.group()
                # Replace comma with dot for parsing
                value_str = value_str.replace(",", ".")
                try:
                    return float(value_str)
                except ValueError:
                    continue

        return None


class MultiValueOCREngine(OCREngine):
    """OCR Engine that can extract multiple numeric values from an image."""

    def extract_all_values(
        self,
        image: np.ndarray,
        preprocessing: str = "auto",
        roi: tuple[int, int, int, int] | None = None,
    ) -> list[OCRResult]:
        """Extract all numeric values from an image.

        Args:
            image: OpenCV image (BGR format)
            preprocessing: Preprocessing method to use
            roi: Region of interest (x, y, width, height), None for full image

        Returns:
            List of OCRResult objects for each found value
        """
        results = []

        try:
            # Crop to ROI if specified
            if roi and roi[2] > 0 and roi[3] > 0:
                x, y, w, h = roi
                img_h, img_w = image.shape[:2]
                x = max(0, min(x, img_w - 1))
                y = max(0, min(y, img_h - 1))
                w = min(w, img_w - x)
                h = min(h, img_h - y)
                image = image[y : y + h, x : x + w]

            # Preprocess
            processed = self._preprocess(image, preprocessing)

            # Get full text with different PSM for multiple words
            config = "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.-"
            raw_text = pytesseract.image_to_string(processed, config=config).strip()

            # Find all numbers
            patterns = [
                r"-?\d+\.\d+",
                r"-?\d+,\d+",
                r"-?\d+",
            ]

            found_values = set()
            for pattern in patterns:
                for match in re.finditer(pattern, raw_text):
                    value_str = match.group().replace(",", ".")
                    try:
                        value = float(value_str)
                        if value not in found_values:
                            found_values.add(value)
                            results.append(
                                OCRResult(
                                    value=round(value, self.decimal_places),
                                    raw_text=value_str,
                                    confidence=0,  # Individual confidence not available
                                    success=True,
                                )
                            )
                    except ValueError:
                        continue

        except Exception as ex:
            _LOGGER.error("Multi-value OCR extraction failed: %s", ex)

        return results
