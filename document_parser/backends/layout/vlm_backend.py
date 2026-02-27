"""
VLM-based layout detection backend using mineru-vl-utils
"""
from typing import Any, Dict, List
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from document_parser.interfaces import LayoutDetectorBackend


class VLMLayoutBackend(LayoutDetectorBackend):
    """
    VLM-based layout detection backend

    Uses Vision Language Model for detecting layout blocks.
    Leverages mineru-vl-utils for VLM integration.
    """

    def __init__(self):
        self.client = None

    def initialize(self, config: Dict) -> None:
        """
        Initialize VLM client

        Args:
            config: Configuration dictionary with keys:
                - vlm_config: Dictionary containing VLM settings
                    - backend: VLM backend type ('http-client', 'transformers', etc.)
                    - server_url: VLM server URL
                    - server_headers: Optional headers
                    - max_concurrency: Max concurrent requests
                    - http_timeout: Request timeout
        """
        from module.parser.mineru_vl_utils import MinerUClient

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

        print(f"VLM layout detector initialized with backend: {vlm_config.get('backend')}")

    def detect(self, image: Image.Image, **kwargs) -> List[Any]:
        """
        Detect layout using VLM

        Args:
            image: PIL Image to process
            **kwargs: Additional parameters (unused for VLM)

        Returns:
            List of ContentBlock objects from mineru-vl-utils
            Each ContentBlock contains:
                - type: Block type ('text', 'title', 'table', etc.)
                - bbox: Normalized bounding box [xmin, ymin, xmax, ymax]
                - angle: Rotation angle (0, 90, 180, 270, or None)
                - content: Optional recognized content
        """
        # Use mineru-vl-utils layout detection
        content_blocks = self.client.layout_detect([image])

        # layout_detect returns a list of lists (one per image)
        # We passed a single image, so return the first list
        return content_blocks[0] if content_blocks else []

    def cleanup(self) -> None:
        """Cleanup VLM client resources"""
        if self.client is not None:
            # MinerUClient doesn't have explicit cleanup, but we can release reference
            self.client = None