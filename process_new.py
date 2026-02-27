"""
PDF Processor - Backward compatible wrapper

This module maintains the original PDFProcessor API while delegating
to the new ConfigurablePDFProcessor under the hood.
"""
from typing import List, Optional

import fitz
import img2pdf

from document_parser import ParserConfig
from document_parser.processor import ConfigurablePDFProcessor


class PDFProcessor:
    """
    Backward compatible PDF processor wrapper

    This class maintains the original API while delegating to
    the new ConfigurablePDFProcessor under the hood.

    Usage (backward compatible):
        >>> processor = PDFProcessor(device="cpu", zoom_factor=3)
        >>> pages = processor.process("document.pdf")

    Usage (new configurable API):
        >>> from document_parser import ParserConfig, ConfigurablePDFProcessor
        >>> config = ParserConfig(device="cuda")
        >>> processor = ConfigurablePDFProcessor(config)
        >>> pages = processor.process("document.pdf")
    """

    def __init__(self, zoom_factor=3, device="cpu", model_source="huggingface", config: ParserConfig = None):
        """
        Initialize PDF processor

        Args:
            zoom_factor: Zoom factor for PDF rendering (default: 3)
            device: Device to use, 'cpu' or 'cuda' (default: 'cpu')
            model_source: Model source for legacy mode (default: 'huggingface')
            config: Optional ParserConfig for new configurable mode

        Note:
            If config is provided, it takes precedence over other parameters.
            If config is None, uses legacy default behavior.
        """
        if config is None:
            # Create default config for backward compatibility
            from document_parser import LayoutConfig, TableConfig
            config = ParserConfig(
                device=device,
                zoom_factor=zoom_factor,
                layout=LayoutConfig(model_source=model_source),
                table=TableConfig(model_source=model_source)
            )

        self._processor = ConfigurablePDFProcessor(config)
        self.zoom_factor = zoom_factor
        self.device = device
        self.model_source = model_source

    def process(self, file_path=None, document=None):
        """
        Process PDF file

        Args:
            file_path: Path to PDF file
            document: fitz.Document object

        Returns:
            List of Page objects
        """
        return self._processor.process(file_path, document)

    def process_img(self, image_path):
        """
        Process image file

        Args:
            image_path: Path to image file

        Returns:
            List of Page objects
        """
        return self._processor.process_img(image_path)

    def set_layout_detector(self, model):
        """
        Set custom layout detector (legacy method)

        Note: This is kept for backward compatibility but may not work
        with the new configurable architecture. Prefer using config instead.
        """
        # Pass through to underlying processor if possible
        if hasattr(self._processor, 'set_layout_detector'):
            self._processor.set_layout_detector(model)

    @staticmethod
    def convert_to_markdown(pages, output_file="list_output.md"):
        """
        Convert pages to markdown

        Args:
            pages: List of Page objects
            output_file: Output file path

        Returns:
            Markdown content string
        """
        from entity.page import Page
        import json

        markdown_lines = []
        for page in pages:
            for block in page.blocks:
                if hasattr(block, 'content') and block.content:
                    markdown_lines.append(block.content)
                    markdown_lines.append("")

        markdown_content = "\n".join(markdown_lines)

        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        return markdown_content

    @staticmethod
    def merge(pages, output_file="json_output.json"):
        """
        Merge pages to JSON

        Args:
            pages: List of Page objects
            output_file: Output file path

        Returns:
            JSON content string
        """
        import json

        pages_data = []
        for page in pages:
            page_dict = {
                'page_num': page.page_num,
                'blocks': []
            }

            for block in page.blocks:
                block_dict = {
                    'block_id': block.block_id,
                    'bbox': [block.x_1, block.y_1, block.x_2, block.y_2],
                    'label': block.label,
                    'content': block.content if hasattr(block, 'content') else ''
                }
                page_dict['blocks'].append(block_dict)

            pages_data.append(page_dict)

        json_content = json.dumps(pages_data, indent=2, ensure_ascii=False)

        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_content)

        return json_content

    @staticmethod
    def pre_process(img, page, is_scanned=False):
        """
        Pre-process image (legacy static method)

        Note: This method is kept for backward compatibility.
        The new configurable processor handles preprocessing internally.
        """
        # Import existing preprocessing logic if needed
        from module.rotation.orientation_corrector import OrientationCorrector

        if is_scanned:
            img_rotated, rotated_angle = OrientationCorrector.correct_orientation(img, page)
            return img_rotated, rotated_angle
        else:
            return img, 0


# For backward compatibility, also expose process_pdf function
def process_pdf(file_path: str, device: str = "cpu", zoom_factor: int = 3):
    """
    Convenience function to process PDF

    Args:
        file_path: Path to PDF file
        device: Device to use
        zoom_factor: Zoom factor

    Returns:
        List of Page objects
    """
    processor = PDFProcessor(device=device, zoom_factor=zoom_factor)
    return processor.process(file_path)