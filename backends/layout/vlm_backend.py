"""
VLM-based layout detection backend for standalone document parser
"""
from typing import Any, Dict, List
from PIL import Image

from ...interfaces import LayoutDetectorBackend


class VLMLayoutBackend(LayoutDetectorBackend):
    """
    VLM-based layout detection backend

    Uses Vision Language Model for detecting layout blocks.
    """

    def __init__(self):
        self.client = None

    def initialize(self, config: Dict) -> None:
        """
        Initialize VLM client

        Args:
            config: Configuration dictionary with keys:
                - vlm_config: Dictionary containing VLM settings
        """
        # Try to import mineru-vl-utils, fallback to simple HTTP client if not available
        try:
            from ...module.parser.mineru_vl_utils import MinerUClient
        except ImportError:
            MinerUClient = None

        vlm_config = config.get('vlm_config', {})

        if MinerUClient:
            self.client = MinerUClient(
                backend=vlm_config.get('backend', 'http-client'),
                server_url=vlm_config.get('server_url', 'http://localhost:8000'),
                server_headers=vlm_config.get('server_headers'),
                max_concurrency=vlm_config.get('max_concurrency', 100),
                http_timeout=vlm_config.get('http_timeout', 600),
                use_tqdm=vlm_config.get('use_tqdm', True),
                debug=vlm_config.get('debug', False),
            )
        else:
            # Use simple HTTP client as fallback
            self.client = SimpleVLMClient(
                server_url=vlm_config.get('server_url', 'http://localhost:8000'),
                server_headers=vlm_config.get('server_headers'),
                http_timeout=vlm_config.get('http_timeout', 600),
            )

        print(f"VLM layout detector initialized with backend: {vlm_config.get('backend', 'http-client')}")

    def detect(self, image: Image.Image, **kwargs) -> List[Any]:
        """
        Detect layout using VLM

        Args:
            image: PIL Image to process
            **kwargs: Additional parameters

        Returns:
            List of ContentBlock objects from VLM
        """
        content_blocks = self.client.layout_detect([image])
        return content_blocks[0] if content_blocks else []

    def cleanup(self) -> None:
        """Cleanup VLM client resources"""
        if self.client is not None:
            self.client = None


class SimpleVLMClient:
    """Simple HTTP client for VLM server as fallback"""

    def __init__(self, server_url: str, server_headers: dict = None, http_timeout: int = 600):
        self.server_url = server_url
        self.headers = server_headers or {}
        self.timeout = http_timeout

    def layout_detect(self, images: List[Image.Image]) -> List[List[Any]]:
        """Detect layout using HTTP API"""
        import base64
        import io
        import requests

        results = []
        for image in images:
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            # Send to VLM server
            response = requests.post(
                f"{self.server_url}/layout",
                json={"image": img_base64},
                headers=self.headers,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                # Convert to ContentBlock-like format
                blocks = []
                for item in result.get('blocks', []):
                    blocks.append(type('ContentBlock', (), {
                        'type': item.get('type', 'text'),
                        'bbox': item.get('bbox', [0, 0, 0, 0]),
                        'content': item.get('content', ''),
                        'score': item.get('score', 1.0)
                    })())
                results.append(blocks)
            else:
                results.append([])

        return results
