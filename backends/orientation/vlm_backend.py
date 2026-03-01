"""
VLM-based orientation detection backend with custom prompts
"""
from typing import Dict, Tuple
from PIL import Image
import fitz
import re

from ...interfaces import OrientationDetectorBackend


class VLMOrientationBackend(OrientationDetectorBackend):
    """
    VLM-based orientation detection backend

    Uses Vision Language Model with custom prompts to identify
    document rotation and correct it.
    """

    def __init__(self):
        self.client = None

    def initialize(self, config: Dict) -> None:
        """
        Initialize VLM backend

        Args:
            config: Configuration dictionary with keys:
                - vlm_config: VLM configuration
        """
        from ...module.parser.mineru_vl_utils import MinerUClient

        vlm_config = config.get('vlm_config', {})

        self.client = MinerUClient(
            backend=vlm_config.get('backend', 'http-client'),
            server_url=vlm_config.get('server_url', 'http://localhost:8000'),
            server_headers=vlm_config.get('server_headers'),
            max_concurrency=vlm_config.get('max_concurrency', 100),
            http_timeout=vlm_config.get('http_timeout', 600),
            use_tqdm=vlm_config.get('use_tqdm', True),
            debug=vlm_config.get('debug', False),
        )

        print("VLM orientation backend initialized")

    def detect(self, image: Image.Image, pdf_page: fitz.Page, **kwargs) -> Tuple[Image.Image, int]:
        """
        Detect and correct orientation using VLM

        Args:
            image: PIL Image to process
            pdf_page: fitz.Page object (for rotation metadata)
            **kwargs: Additional parameters

        Returns:
            Tuple of (rotated_image, rotation_angle)
        """
        # Create prompt to ask VLM about orientation
        prompt = """Analyze this document image and determine its orientation.

The image may be rotated. Determine if it is:
- 0 degrees (upright, normal orientation)
- 90 degrees (rotated clockwise)
- 180 degrees (upside down)
- 270 degrees (rotated counter-clockwise)

Respond with ONLY the degree value (0, 90, 180, or 270). Do not include any other text."""

        try:
            # Use VLM client to analyze the image
            # Most VLM APIs accept image and text prompt
            # The exact API depends on the mineru-vl-utils implementation

            # For http-client backend, we can use the raw chat completion
            from ...module.parser.mineru_vl_utils.vlm_client.http_client import HttpClient

            if isinstance(self.client.client, HttpClient):
                # Get response from VLM
                response = self.client.client.chat_completion(
                    images=[image],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10
                )

                # Extract angle from response
                response_text = response.get('choices', [{}])[0].get('message', {}).get('content', '0')

                # Parse the angle
                angle_match = re.search(r'\b(0|90|180|270)\b', response_text)
                if angle_match:
                    detected_angle = int(angle_match.group(1))
                else:
                    # Default to 0 if parsing fails
                    detected_angle = 0
            else:
                # For other backends, use simple heuristic
                detected_angle = 0

            # Rotate image to correct orientation
            if detected_angle != 0:
                # PIL rotates counter-clockwise, so we need to rotate clockwise
                # to correct: 90° CW = 270° CCW, 180° CW = 180° CCW, 270° CW = 90° CCW
                rotation_map = {90: 270, 180: 180, 270: 90}
                pil_angle = rotation_map[detected_angle]
                rotated_image = image.rotate(pil_angle, expand=True)
            else:
                rotated_image = image

            return rotated_image, detected_angle

        except Exception as e:
            print(f"Warning: VLM orientation detection failed: {e}")
            # Fallback: return original image
            return image, 0

    def cleanup(self) -> None:
        """Cleanup VLM client"""
        if self.client is not None:
            self.client = None