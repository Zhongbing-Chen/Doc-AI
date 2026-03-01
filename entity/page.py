"""Page class for document parser"""
import math
from dataclasses import dataclass
from typing import Union, List
from pathlib import Path

import fitz
import numpy as np
import torch

from .block import Block, TableStructure
from .box import Box, OcrBlock, BoxUtil


@dataclass
class Page:
    """
    Page class representing a document page.

    Attributes:
        page_num: Page number
        blocks: List of blocks in the page
        rotated_angle: Rotation angle detected
        skewed_angle: Skew angle detected
        pdf_page: fitz.Page object
        image: PIL Image of the page
        zoom_factor: Zoom factor used
        is_scanned: Whether page is scanned
        ocr_blocks: OCR recognition results
    """
    page_num: Union[None, int]
    blocks: Union[None, List[Block]]
    rotated_angle: Union[None, float]
    skewed_angle: Union[None, float]
    pdf_page: fitz.Page
    image: Union[str, Path, int, list, tuple, np.ndarray]
    zoom_factor: float
    is_scanned: bool
    ocr_blocks: List[OcrBlock]
    raw_ocr_result: list = None

    def __init__(self, pdf_page, page_num, image, zoom_factor, items=None, rotated_angle=None, skewed_angle=None,
                 is_scanned=False):
        self.pdf_page = pdf_page
        self.page_num = page_num
        self.blocks = items
        self.rotated_angle = rotated_angle
        self.image = image
        self.zoom_factor = zoom_factor
        self.is_scanned = is_scanned
        self.skewed_angle = skewed_angle
        self.ocr_blocks = []
        self.raw_ocr_result = None

        # Initialize OCR results if image is available
        if self.image is not None:
            try:
                from ..module.text.text_parser import TextExtractor
                if self.is_scanned:
                    self.raw_ocr_result = TextExtractor.ocr_all_image_result(self.image)
                    self.ocr_blocks = OcrBlock.from_rapid_ocr(self.raw_ocr_result)
                else:
                    self.raw_ocr_result = TextExtractor.detection_all_image_result(self.image)
                    self.ocr_blocks = OcrBlock.from_rapid_ocr(self.raw_ocr_result)
            except ImportError:
                # TextExtractor not available, skip OCR
                pass

    def build_blocks(self, results):
        """Build blocks from YOLO results"""
        self.blocks = [Block.from_bbox(i, self.page_num, box) for i, box in
                       enumerate(results.boxes.data)]

    @classmethod
    def build_blocks_with_results(cls, results):
        """Build blocks from YOLO results (class method)"""
        return [Block.from_bbox(i, "", box) for i, box in
                enumerate(results.boxes.data)]

    def build_blocks_from_vlm(self, content_blocks: List) -> None:
        """
        Build blocks from VLM ContentBlocks (mineru-vl-utils format)

        Args:
            content_blocks: List of ContentBlock objects from VLM
        """
        self.blocks = []
        for idx, cb in enumerate(content_blocks):
            # Convert normalized bbox to pixel coordinates
            xmin, ymin, xmax, ymax = cb.bbox
            block = Block(
                block_id=idx,
                x_1=xmin * self.image.width,
                y_1=ymin * self.image.height,
                x_2=xmax * self.image.width,
                y_2=ymax * self.image.height,
                label=self._get_label_from_type(cb.type),
                label_id=self._get_label_id_from_type(cb.type),
                page_num=self.page_num,
                layout_score=getattr(cb, 'score', 1.0)
            )
            # Store content if available
            if hasattr(cb, 'content') and cb.content:
                block.content = cb.content
            self.blocks.append(block)

    @staticmethod
    def _get_label_from_type(type_str: str) -> str:
        """Convert VLM type to Block label"""
        type_map = {
            'text': 'Text',
            'title': 'Title',
            'table': 'Table',
            'figure': 'Figure',
            'equation': 'Equation',
            'header': 'Header',
            'footer': 'Footer',
            'toc': 'Toc',
            'figure_caption': 'Figure caption',
            'table_caption': 'Table caption'
        }
        return type_map.get(type_str.lower(), 'Text')

    @staticmethod
    def _get_label_id_from_type(type_str: str) -> int:
        """Convert VLM type to Block label_id"""
        type_map = {
            'text': 0,
            'title': 1,
            'header': 2,
            'footer': 3,
            'figure': 4,
            'table': 5,
            'toc': 6,
            'figure_caption': 7,
            'table_caption': 8
        }
        return type_map.get(type_str.lower(), 0)

    def extract_text_with_backend(self, text_backend, is_scanned: bool = None) -> None:
        """
        Extract text using the provided text backend

        Args:
            text_backend: TextExtractorBackend instance
            is_scanned: Whether the page is scanned (auto-detected if None)
        """
        if is_scanned is None:
            is_scanned = self.is_scanned

        for block in self.blocks:
            # Get bbox in image coordinates
            bbox = [block.x_1, block.y_1, block.x_2, block.y_2]

            # Extract text using backend
            block.content = text_backend.extract(
                image=self.image,
                pdf_page=self.pdf_page,
                bbox=bbox,
                is_scanned=is_scanned
            )

            # Recognize table content if block is a table
            if block.label == "Table" and block.table_structure:
                self.recognize_table_content(block, self.pdf_page, self.zoom_factor, is_scanned)

    def extract_text(self):
        """Extract text from the items using fitz API"""
        from ..module.text.text_parser import TextExtractor
        from .box import BoxUtil

        if not self.is_scanned:
            for block in self.blocks:
                block.content = TextExtractor.parse_by_fitz(self.pdf_page, block.adjusted_bbox_with_zoom(self.zoom_factor))
                self.recognize_table_content(block, self.pdf_page, self.zoom_factor, self.is_scanned)
        else:
            self.ocr_blocks = OcrBlock.from_rapid_ocr(self.raw_ocr_result)
            BoxUtil.match_box_to_ocr(self.blocks, self.ocr_blocks)

            for block in self.blocks:
                self.recognize_table_content(block, self.pdf_page, self.zoom_factor, self.is_scanned, self.ocr_blocks)

    @classmethod
    def recognize_table_content(cls, block: Block, pdf_page, zoom_factor, is_scanned: bool,
                                ocr_blocks: List[OcrBlock] = None) -> None:
        """Recognize table content"""
        from ..module.text.text_parser import TextExtractor
        from .box import BoxUtil

        if block.label == "Table":
            print("Recognize Table Content")

            if not block.table_structure:
                return

            if is_scanned:
                BoxUtil.match_box_to_ocr(block.table_structure, ocr_blocks)
            else:
                for cell in block.table_structure:
                    cell.content = TextExtractor.parse_by_fitz(pdf_page, [coord / zoom_factor for coord in cell.bbox])

    def recognize_table(self, table_parser, save_path: str):
        """Recognize table structure"""
        for block in self.blocks:
            block.recognize_table_structure(self.image, table_parser, save_path)

    @property
    def texts(self):
        """Get list of text content from blocks"""
        return [item.markdown_content for item in self.blocks]

    def sort(self):
        """Sort items based on bounding box"""
        if len(self.blocks) == 0:
            return

        try:
            from ..util.gap_tree import GapTree
            gtree = GapTree(lambda item: item.bbox)
            sorted_text_blocks = gtree.sort(self.blocks)
        except ImportError:
            # Simple sort if gap_tree not available
            sorted_text_blocks = sorted(self.blocks, key=lambda x: (x.y_1 // 10 * 10, x.x_1))

        for i, item in enumerate(sorted_text_blocks):
            item.block_id = i

        self.blocks = sorted_text_blocks

    def filter_items_by_label(self, filters=None) -> None:
        """Filter items by label"""
        if filters is None:
            return
        items = []
        for item in self.blocks:
            if item.label not in filters:
                items.append(item)
        self.blocks = items

    def merge_overlap_block(self, threshold=0.8) -> None:
        """Merge overlapping blocks based on IOB algorithm"""
        from ..util.coordinate_util import CoordinateUtil

        i = 0
        while i < len(self.blocks) - 1:
            cur_block: Block = self.blocks[i]
            j = i + 1
            while j < len(self.blocks):
                next_block: Block = self.blocks[j]
                ratio_a, ratio_b = CoordinateUtil.iob(cur_block.bbox, next_block.bbox)
                if ratio_a >= threshold or ratio_b >= threshold:
                    cur_block.union(next_block)
                    del self.blocks[j]
                else:
                    j += 1
            i += 1

    def add_text_layer(self) -> None:
        """Write OCR results to PDF with appropriate font sizes"""
        if not self.is_scanned:
            return

        for item in self.raw_ocr_result:
            bbox, text, confidence = item
            font_size = self.calculate_font_size(bbox, text)
            rect = fitz.Rect(bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1])
            text_kwargs = {
                "fontname": "helvetica",
                "fontsize": font_size
            }
            self.pdf_page.insert_text(rect.tl, text, **text_kwargs)

    @staticmethod
    def calculate_font_size(bbox, text):
        """Calculate font size based on bounding box and text length"""
        width = math.sqrt((bbox[1][0] - bbox[0][0]) ** 2 + (bbox[1][1] - bbox[0][1]) ** 2)
        height = math.sqrt((bbox[3][0] - bbox[0][0]) ** 2 + (bbox[3][1] - bbox[0][1]) ** 2)

        width_pt = width * 72 / 96
        height_pt = height * 72 / 96

        font_size_by_height = height_pt * 0.7

        char_count = len(text)
        font_size_by_width = width_pt / (char_count * 0.6)

        return min(font_size_by_height, font_size_by_width)

    def to_map(self):
        """Convert page to map format"""
        return [item.to_json() for item in self.blocks]

    def fix_block_using_ocr(self):
        """Fix blocks using OCR results"""
        from ..module.layout.layout_constant import TYPE_DICT

        for ocr_block in self.ocr_blocks:
            for block in self.blocks:
                overlap_area = BoxUtil.calculate_overlap_area(block, ocr_block)
                ocr_area = BoxUtil.calculate_box_area(ocr_block)

                if ocr_area > 0 and overlap_area > 0:
                    ocr_block.relationships[block.block_id] = overlap_area / ocr_area

        for ocr_block in self.ocr_blocks:
            if ocr_block.relationships:
                max_block_id = max(ocr_block.relationships, key=ocr_block.relationships.get)
                for block in self.blocks:
                    if block.block_id == max_block_id:
                        if ocr_block.relationships[max_block_id] < 0.9:
                            BoxUtil.extend_layout_box(block, ocr_block)
            else:
                label_id = self.get_key_from_value(TYPE_DICT, "Text")
                layout = Block(
                    block_id="",
                    x_1=ocr_block.x_1,
                    y_1=ocr_block.y_1,
                    x_2=ocr_block.x_2,
                    y_2=ocr_block.y_2,
                    label="Text",
                    label_id=label_id,
                    page_num=self.page_num,
                    layout_score=1.0,
                )
                self.blocks.append(layout)

    @staticmethod
    def get_key_from_value(dictionary, value):
        """Get key from value in dictionary"""
        for key, val in dictionary.items():
            if val == value:
                return key
        return 0

    def compress_copy(self) -> 'Page':
        """Create a compressed copy of the page"""
        copy = Page(
            pdf_page=self.pdf_page,
            page_num=self.page_num,
            image=None,  # Don't copy image to save memory
            zoom_factor=self.zoom_factor,
            items=self.blocks,
            rotated_angle=self.rotated_angle,
            skewed_angle=self.skewed_angle,
            is_scanned=self.is_scanned
        )
        copy.ocr_blocks = self.ocr_blocks
        copy.raw_ocr_result = self.raw_ocr_result
        return copy
