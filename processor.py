"""
Configurable PDF Processor with pluggable backends

This is the main processor that uses the configuration system and
backend architecture to provide flexible document parsing.
"""
import io
import os
import time
from datetime import datetime
from typing import List, Optional

import fitz
import img2pdf
from PIL import Image

from .config import ParserConfig
from .backend_factory import BackendFactory
from .entity.page import Page
from .util.visualizer import Visualizer


class ConfigurablePDFProcessor:
    """
    Configurable PDF processor with pluggable backends

    This processor allows each processing step to use different backends
    (YOLO, VLM, OCR, etc.) based on configuration.
    """

    def __init__(self, config: Optional[ParserConfig] = None):
        """
        Initialize processor with configuration

        Args:
            config: ParserConfig object. If None, uses default configuration.
        """
        self.config = config or ParserConfig.default()
        self._initialize_backends()

    def _initialize_backends(self) -> None:
        """Initialize all processing backends from configuration"""
        cfg = self.config

        # Prepare VLM config for backends that need it
        vlm_config = cfg.vlm.to_dict()

        # Orientation backend
        if cfg.orientation.enabled:
            self.orientation_backend = BackendFactory.create_orientation_backend(
                cfg.orientation.backend,
                {
                    **cfg.orientation.to_dict(),
                    'vlm_config': vlm_config
                }
            )
        else:
            self.orientation_backend = None

        # Skew backend
        if cfg.skew.enabled:
            self.skew_backend = BackendFactory.create_skew_backend(
                cfg.skew.backend,
                cfg.skew.to_dict()
            )
        else:
            self.skew_backend = None

        # Layout backend
        if cfg.layout.enabled:
            self.layout_backend = BackendFactory.create_layout_backend(
                cfg.layout.backend,
                {
                    **cfg.layout.to_dict(),
                    'device': cfg.device,
                    'vlm_config': vlm_config
                }
            )
        else:
            self.layout_backend = None

        # Table backend
        if cfg.table.enabled:
            self.table_backend = BackendFactory.create_table_backend(
                cfg.table.backend,
                {
                    **cfg.table.to_dict(),
                    'device': cfg.device,
                    'vlm_config': vlm_config
                }
            )
        else:
            self.table_backend = None

        # Text backend
        if cfg.text.enabled:
            self.text_backend = BackendFactory.create_text_backend(
                cfg.text.backend,
                {
                    **cfg.text.to_dict(),
                    'vlm_config': vlm_config
                }
            )
        else:
            self.text_backend = None

    def process(self, file_path: Optional[str] = None, document=None) -> List[Page]:
        """
        Process PDF file or document

        Args:
            file_path: Path to PDF file
            document: fitz.Document object

        Returns:
            List of Page objects with extracted content
        """
        pages = []

        # Create output directories
        current_time_string = datetime.now().strftime("%Y%m%d%H%M%S")
        dir_path_detail = f"{self.config.output_dir}/{current_time_string}/detail"
        dir_path_table = f"{self.config.output_dir}/{current_time_string}/table"
        os.makedirs(dir_path_detail, exist_ok=True)
        os.makedirs(dir_path_table, exist_ok=True)

        # Open document
        doc = document if file_path is None else fitz.open(file_path)

        # Process each page
        for page_num in range(len(doc)):
            processed_page = self.process_one_page(dir_path_table, doc, page_num)
            compressed_page = processed_page.compress_copy()
            pages.append(compressed_page)

        # Visualization
        if self.config.visualization:
            Visualizer.depict_bbox(pages, dir_path_detail)

        doc.close()
        return pages

    def process_one_page(self, dir_path_table: str, doc, page_num: int) -> Page:
        """
        Process a single page

        Args:
            dir_path_table: Directory for table outputs
            doc: fitz.Document object
            page_num: Page number (0-indexed)

        Returns:
            Page object with extracted content
        """
        print(f"Processing page: {page_num}")
        start = time.time()

        page = doc[page_num]

        # Step 1: Convert to image
        img, img_bytes = self.convert_page_to_image(page)
        img_rotated = img

        # Step 2: Determine if scanned (for text extraction)
        is_scanned = self.text_backend.is_scanned(page) if self.text_backend else False

        # Step 3: Orientation detection
        rotated_angle = 0
        if self.orientation_backend:
            img_rotated, rotated_angle = self.orientation_backend.detect(img_rotated, page)

        # Step 4: Skew detection
        deskew_angle = 0
        if self.skew_backend and is_scanned:
            img_rotated, deskew_angle = self.skew_backend.detect_and_correct(img_rotated)

        # Step 5: Create Page object
        page_obj = Page(
            page_num=page_num,
            image=img_rotated,
            pdf_page=page,
            zoom_factor=self.config.zoom_factor,
            rotated_angle=rotated_angle,
            skewed_angle=deskew_angle,
            is_scanned=is_scanned
        )

        # Step 6: Layout detection
        if self.layout_backend:
            layout = self.layout_backend.detect(
                page_obj.image,
                conf=self.config.layout.confidence,
                iou=self.config.layout.iou_threshold
            )

            # Build blocks from layout results
            if layout is not None:
                # Check if it's YOLO results or VLM ContentBlocks
                if hasattr(layout, 'boxes'):
                    # YOLO results
                    page_obj.build_blocks(layout)
                    page_obj.fix_block_using_ocr()
                    page_obj.filter_items_by_label(filters=["Header", "Footer", "Figure"])
                    page_obj.merge_overlap_block(threshold=0.7)
                    page_obj.sort()
                else:
                    # VLM ContentBlocks - need different handling
                    page_obj.build_blocks_from_vlm(layout)
                    page_obj.sort()

        # Step 7: Table parsing
        if self.table_backend:
            page_obj.recognize_table(self.table_backend, dir_path_table)

        # Step 8: Text extraction
        if self.text_backend:
            page_obj.extract_text_with_backend(self.text_backend, is_scanned)
            page_obj.add_text_layer()

        print(f"Time taken for page {page_num}: {time.time() - start:.2f}s")
        return page_obj

    def convert_page_to_image(self, page):
        """
        Convert PDF page to PIL Image

        Args:
            page: fitz.Page object

        Returns:
            Tuple of (PIL Image, image bytes)
        """
        pix = page.get_pixmap(matrix=fitz.Matrix(self.config.zoom_factor, self.config.zoom_factor))
        img_bytes = pix.tobytes("ppm")
        img_stream = io.BytesIO(img_bytes)
        img = Image.open(img_stream)
        return img, img_bytes

    def process_img(self, image_path: str) -> List[Page]:
        """
        Process image file

        Args:
            image_path: Path to image file

        Returns:
            List of Page objects (typically 1)
        """
        pdf_bytes = img2pdf.convert(image_path)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return self.process(document=doc)

    @staticmethod
    def convert_to_markdown(pages: List[Page], output_file: str = "list_output.md"):
        """
        Convert pages to markdown

        Args:
            pages: List of Page objects
            output_file: Output file path

        Returns:
            Markdown content string
        """
        from .export import convert_to_markdown
        return convert_to_markdown(pages, output_file)

    @staticmethod
    def merge(pages: List[Page], output_file: str = "json_output.json"):
        """
        Merge pages to JSON

        Args:
            pages: List of Page objects
            output_file: Output file path

        Returns:
            JSON content string
        """
        from .export import merge
        return merge(pages, output_file)