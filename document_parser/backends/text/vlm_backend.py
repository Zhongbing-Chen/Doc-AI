"""
VLM-based text extraction backend
"""
from typing import Dict, Optional, List
from PIL import Image
import fitz

from document_parser.interfaces import TextExtractorBackend


class VLMTextBackend(TextExtractorBackend):
    """
    VLM-based text extraction backend

    Uses Vision Language Model for text extraction.
    Works well for both digital and scanned documents.
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

        print(f"VLM text backend initialized")

    def extract(self, image: Image.Image, bbox: Optional[List[float]] = None, **kwargs) -> str:
        """
        Extract text using VLM

        Args:
            image: PIL Image to process
            bbox: Optional bounding box [x1, y1, x2, y2] to crop
            **kwargs: Additional parameters

        Returns:
            Extracted text string
        """
        # Crop image if bbox provided
        if bbox:
            image = image.crop(bbox)

        # Use VLM for text extraction
        # The two_step_extract method returns ContentBlock objects
        content_blocks = self.client.two_step_extract([image])

        if not content_blocks or not content_blocks[0]:
            return ""

        # Extract text from all text blocks
        text_parts = []
        for block in content_blocks[0]:
            if hasattr(block, 'type') and block.type == 'text':
                if hasattr(block, 'content') and block.content:
                    text_parts.append(block.content)

        return ' '.join(text_parts)

    def is_scanned(self, pdf_page: fitz.Page) -> bool:
        """
        VLM can handle both scanned and digital

        Args:
            pdf_page: fitz.Page object

        Returns:
            True (VLM works for both types)
        """
        # VLM can handle both, but we still check for other logic
        page_text = pdf_page.get_text()
        return len(page_text.strip()) == 0

    def cleanup(self) -> None:
        """Cleanup VLM client"""
        if self.client is not None:
            self.client = None