"""
VLM-based table parsing backend
"""
from typing import Dict, Optional, List
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from document_parser.interfaces import TableParserBackend
from entity.block import TableStructure


class VLMTableBackend(TableParserBackend):
    """
    VLM-based table parsing backend

    Uses Vision Language Model for table structure recognition.
    Can handle complex tables with merged cells.
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

        print("VLM table backend initialized")

    def parse(self, image: Image.Image, bbox: Optional[List[float]] = None, **kwargs) -> List[TableStructure]:
        """
        Parse table using VLM

        Args:
            image: PIL Image containing the table
            bbox: Optional bounding box [x1, y1, x2, y2] to crop
            **kwargs: Additional parameters

        Returns:
            List of TableStructure objects
        """
        from module.parser.mineru_vl_utils.helpers import ExtractHelper

        # Crop image if bbox provided
        if bbox:
            image = image.crop(bbox)

        # Use VLM for table extraction
        content_blocks = self.client.two_step_extract([image])

        if not content_blocks or not content_blocks[0]:
            return []

        # Find table block
        table_block = None
        for block in content_blocks[0]:
            if hasattr(block, 'type') and block.type == 'table':
                table_block = block
                break

        if not table_block:
            return []

        # Convert HTML to TableStructure
        # Use the ExtractHelper from mineru-vl-utils
        helper = ExtractHelper()

        # If table_block has content (HTML), convert to TableStructure
        if hasattr(table_block, 'content') and table_block.content:
            # The content is HTML, we need to parse it
            # For now, return empty list as HTML->TableStructure conversion
            # would require additional parsing logic
            # TODO: Implement HTML to TableStructure conversion
            return []

        return []

    def cleanup(self) -> None:
        """Cleanup VLM client"""
        if self.client is not None:
            self.client = None