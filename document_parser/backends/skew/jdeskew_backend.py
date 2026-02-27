"""
Jdeskew backend for skew detection and correction
"""
from typing import Dict, Tuple
from PIL import Image
import numpy as np

from document_parser.interfaces import SkewDetectorBackend


class JdeskewBackend(SkewDetectorBackend):
    """
    Jdeskew backend for skew detection and correction

    Uses the jdeskew library to detect and correct small angle
    rotations in scanned documents.
    """

    def __init__(self):
        self.angle_max = 10.0
        self.threshold = 0.3

    def initialize(self, config: Dict) -> None:
        """
        Initialize Jdeskew backend

        Args:
            config: Configuration dictionary with keys:
                - angle_max: Maximum angle to correct (default: 10.0)
                - threshold: Threshold for skew detection (default: 0.3)
        """
        self.angle_max = config.get('angle_max', 10.0)
        self.threshold = config.get('threshold', 0.3)
        print(f"Jdeskew backend initialized (max angle: {self.angle_max})")

    def detect_and_correct(self, image: Image.Image, **kwargs) -> Tuple[Image.Image, float]:
        """
        Detect and correct skew

        Args:
            image: PIL Image to process
            **kwargs: Additional parameters

        Returns:
            Tuple of (deskewed_image, skew_angle)
        """
        from jdeskew.estimator import get_deskew_angle
        from jdeskew.utility import apply_deskew

        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Estimate skew angle
        skew_angle = get_deskew_angle(img_array)

        # Limit to max angle
        if abs(skew_angle) > self.angle_max:
            skew_angle = 0.0

        # Correct skew if angle is significant
        if abs(skew_angle) > self.threshold:
            deskewed_array = apply_deskew(img_array, skew_angle)
            deskewed_image = Image.fromarray(deskewed_array)
        else:
            deskewed_image = image
            skew_angle = 0.0

        return deskewed_image, skew_angle

    def cleanup(self) -> None:
        """No cleanup needed"""
        pass