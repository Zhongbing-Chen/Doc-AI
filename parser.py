"""
Document Parser - Main Parser Module

Parses documents (PDF, Word, Excel, PPT, Markdown, Images) into uniform JSON format.
Uses VLM for complete document understanding when available.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List
from PIL import Image

try:
    from .format import (
        Document,
        DocumentMetadata,
        DocumentInfo,
        Page,
        Block,
        BBox,
        Content,
        TableCell,
        BlockType,
        DocumentType,
    )
    from .universal_parser import UniversalDocumentParser
except ImportError:
    from format import (
        Document,
        DocumentMetadata,
        DocumentInfo,
        Page,
        Block,
        BBox,
        Content,
        TableCell,
        BlockType,
        DocumentType,
    )
    from universal_parser import UniversalDocumentParser

try:
    import httpx
except ImportError:
    httpx = None


class DocumentParser:
    """
    Main document parser that converts any supported document format
    into the uniform JSON structure.

    Supports:
    - PDF (with VLM for full analysis)
    - Word (.doc, .docx)
    - Excel (.xls, .xlsx)
    - PowerPoint (.ppt, .pptx)
    - Markdown (.md, .markdown)
    - Images (.png, .jpg, .jpeg)
    """

    MIME_TYPES = {
        '.pdf': 'application/pdf',
        '.doc': 'application/msword',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.xls': 'application/vnd.ms-excel',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.ppt': 'application/vnd.ms-powerpoint',
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        '.md': 'text/markdown',
        '.markdown': 'text/markdown',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
    }

    def __init__(self, vlm_url: Optional[str] = None):
        """
        Initialize the document parser.

        Args:
            vlm_url: VLM server URL for VLM-based parsing.
                     If provided, VLM will be used for layout detection,
                     text extraction, and table parsing.
        """
        self.vlm_url = vlm_url
        self.parser = UniversalDocumentParser(
            enable_vlm=(vlm_url is not None),
            vlm_url=vlm_url
        )

    def parse(self, file_path: Union[str, Path]) -> Document:
        """
        Parse a document file into uniform JSON format.

        Args:
            file_path: Path to the document file

        Returns:
            Document object with uniform JSON structure
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = file_path.suffix.lower()
        file_size = file_path.stat().st_size
        mime_type = self.MIME_TYPES.get(file_ext, 'application/octet-stream')
        doc_type = self._get_document_type(file_ext)

        # Create document structure
        doc = Document()

        # Set metadata
        doc.metadata = DocumentMetadata(
            file_name=file_path.name,
            file_path=str(file_path.absolute()),
            file_size=file_size,
            file_type=doc_type.value,
            file_extension=file_ext,
            mime_type=mime_type,
            processed_at=datetime.utcnow().isoformat() + "Z"
        )

        # Parse based on type
        if doc_type == DocumentType.PDF:
            self._parse_pdf(doc, file_path)
        elif doc_type == DocumentType.WORD:
            self._parse_word(doc, file_path)
        elif doc_type == DocumentType.EXCEL:
            self._parse_excel(doc, file_path)
        elif doc_type == DocumentType.POWERPOINT:
            self._parse_powerpoint(doc, file_path)
        elif doc_type == DocumentType.MARKDOWN:
            self._parse_markdown(doc, file_path)
        elif doc_type == DocumentType.IMAGE:
            self._parse_image(doc, file_path)
        else:
            self._parse_generic(doc, file_path)

        return doc

    def parse_to_json(self, file_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Parse document and return JSON string.

        Args:
            file_path: Path to document
            output_path: Optional path to save JSON file

        Returns:
            JSON string
        """
        doc = self.parse(file_path)
        json_str = doc.to_json()
        if output_path:
            doc.save(output_path)
        return json_str

    def _get_document_type(self, file_ext: str) -> DocumentType:
        """Get document type from extension"""
        mapping = {
            '.pdf': DocumentType.PDF,
            '.doc': DocumentType.WORD,
            '.docx': DocumentType.WORD,
            '.xls': DocumentType.EXCEL,
            '.xlsx': DocumentType.EXCEL,
            '.ppt': DocumentType.POWERPOINT,
            '.pptx': DocumentType.POWERPOINT,
            '.md': DocumentType.MARKDOWN,
            '.markdown': DocumentType.MARKDOWN,
            '.png': DocumentType.IMAGE,
            '.jpg': DocumentType.IMAGE,
            '.jpeg': DocumentType.IMAGE,
        }
        return mapping.get(file_ext.lower(), DocumentType.WORD)

    def _parse_pdf(self, doc: Document, file_path: Path) -> None:
        """
        Parse PDF file.

        If VLM is configured, uses VLM for complete analysis.
        Otherwise uses simple text extraction with fitz.
        """
        if self.vlm_url:
            # Use VLM-based parsing
            self._parse_pdf_with_vlm(doc, file_path)
        else:
            # Use simple fitz-based text extraction
            self._parse_pdf_simple(doc, file_path)

    def _parse_pdf_simple(self, doc: Document, file_path: Path) -> None:
        """
        Parse PDF using simple text extraction (fitz).

        This extracts text and creates basic block structure without
        requiring YOLO or other models.
        """
        import fitz

        pdf_doc = fitz.open(file_path)
        doc.document_info.total_pages = len(pdf_doc)

        full_text_parts = []
        full_markdown_parts = []

        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]

            # Create page structure
            page_rect = page.rect
            page_doc = Page(
                page_num=page_num + 1,
                width=page_rect.width,
                height=page_rect.height,
                page_content=""
            )

            # Extract text blocks
            block_id = 0
            page_text_parts = []

            # Get text blocks with position
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if block.get("type") == 1:  # Image block
                    continue

                block_id += 1
                block_text = ""
                block_bbox = BBox(
                    x_1=block.get("bbox", [0, 0, 0, 0])[0],
                    y_1=block.get("bbox", [0, 0, 0, 0])[1],
                    x_2=block.get("bbox", [0, 0, 0, 0])[2],
                    y_2=block.get("bbox", [0, 0, 0, 0])[3]
                )

                # Extract text from lines
                lines_text = []
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        for span in line.get("spans", []):
                            line_text += span.get("text", "")
                        lines_text.append(line_text)

                block_text = "\n".join(lines_text)
                if block_text.strip():
                    page_doc.blocks.append(Block(
                        block_id=f"p{page_num + 1}_b{block_id:03d}",
                        type=BlockType.PARAGRAPH.value,
                        bbox=block_bbox,
                        content=Content(text=block_text, markdown=block_text),
                        confidence=1.0
                    ))
                    page_text_parts.append(block_text)

            page_doc.page_content = "\n\n".join(page_text_parts)
            doc.pages.append(page_doc)
            full_text_parts.append(page_doc.page_content)
            full_markdown_parts.append(page_doc.page_content)

        doc.full_text = "\n\n".join(full_text_parts)
        doc.full_markdown = doc.full_text
        pdf_doc.close()

    def _parse_pdf_with_vlm(self, doc: Document, file_path: Path) -> None:
        """
        Parse PDF using VLM (vLLM) for complete analysis.

        Uses multimodal VLM served via vLLM on port 8000.
        The VLM analyzes each page and returns structured content including:
        - Layout blocks with bounding boxes
        - Text content
        - Table structures
        - Figure descriptions
        """
        import fitz
        import base64

        pdf_doc = fitz.open(file_path)
        doc.document_info.total_pages = len(pdf_doc)

        full_markdown_parts = []

        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]

            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("png")

            # Call vLLM VLM API for analysis
            vlm_result = self._call_vllm_api(img_bytes, page_num + 1)

            # Create page structure from VLM result
            page_doc = Page(
                page_num=page_num + 1,
                width=pix.width,
                height=pix.height,
                page_content=vlm_result.get("text", "")
            )

            # Parse blocks from VLM result
            if "blocks" in vlm_result:
                page_doc.blocks = self._parse_vlm_blocks(vlm_result["blocks"], page_num + 1)

            doc.pages.append(page_doc)
            full_markdown_parts.append(vlm_result.get("markdown", vlm_result.get("text", "")))

        doc.full_markdown = "\n\n".join(full_markdown_parts)
        doc.full_text = doc.full_markdown
        pdf_doc.close()

    def _call_vllm_api(self, image_bytes: bytes, page_num: int) -> dict:
        """
        Call vLLM VLM API to analyze a page.

        vLLM serves multimodal models (like LLaVA) with OpenAI-compatible API.
        Default endpoint: http://localhost:8000/v1/chat/completions

        Returns structured result with blocks, text, and markdown.
        """
        # Encode image to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # VLM prompt for document analysis
        prompt = """Analyze this document page and extract all content in JSON format.

Extract:
1. All text blocks with their positions (bounding box coordinates)
2. Content type for each block (title, heading_1, heading_2, paragraph, table, figure, equation, etc.)
3. Full text content as plain text
4. Markdown formatted content with proper headings and formatting

Respond ONLY with valid JSON in this exact format:
{
  "text": "full plain text of the page",
  "markdown": "# Heading\\n\\nContent in **markdown** format",
  "blocks": [
    {
      "type": "title|heading_1|heading_2|paragraph|table|figure|equation|list",
      "bbox": {"x_1": 0, "y_1": 0, "x_2": 0, "y_2": 0},
      "content": "block content text"
    }
  ]
}"""

        # vLLM API endpoint (OpenAI-compatible)
        url = f"{self.vlm_url}/v1/chat/completions"

        # vLLM / vLLM OpenAI-compatible API payload
        payload = {
            "model": "default",  # vLLM uses loaded model
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.1  # Low temperature for structured output
        }

        try:
            if httpx is None:
                raise ImportError("httpx not available")
            with httpx.Client(timeout=300.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                # Parse JSON from response
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:-3]
                elif content.startswith("```"):
                    content = content[3:-3]

                return json.loads(content.strip())
        except Exception as e:
            # Fallback: return text-only result
            return {
                "text": f"[VLM parsing error: {str(e)}]",
                "markdown": f"[VLM parsing error: {str(e)}]",
                "blocks": []
            }

    def _parse_vlm_blocks(self, blocks: list, page_num: int) -> List[Block]:
        """Parse VLM block results into our Block format"""
        result = []
        for i, block_data in enumerate(blocks):
            block_id = f"p{page_num}_b{i:03d}"
            block_type = block_data.get("type", "text")

            # Parse bbox
            bbox_data = block_data.get("bbox", {})
            bbox = BBox(
                x_1=bbox_data.get("x_1", 0),
                y_1=bbox_data.get("y_1", 0),
                x_2=bbox_data.get("x_2", 0),
                y_2=bbox_data.get("y_2", 0)
            )

            # Parse content
            content_text = block_data.get("content", "")
            content = Content(text=content_text, markdown=content_text)

            block = Block(
                block_id=block_id,
                type=block_type,
                bbox=bbox,
                content=content,
                confidence=block_data.get("confidence", 0.9)
            )
            result.append(block)

        return result

    def _convert_pdf_page_to_doc(self, page) -> Page:
        """Convert a processed PDF page to our Page format"""
        page_doc = Page(
            page_num=page.page_num + 1 if isinstance(page.page_num, int) else 1,
            width=612,
            height=792,
            page_content=""
        )

        if page.blocks:
            for block in page.blocks:
                content = Content(
                    text=block.content or "",
                    markdown=block.markdown_content or ""
                )

                # Map block label to BlockType
                block_type = self._map_label_to_type(block.label)

                b = Block(
                    block_id=f"p{page_doc.page_num}_b{block.block_id}",
                    type=block_type,
                    bbox=BBox(
                        x_1=block.x_1,
                        y_1=block.y_1,
                        x_2=block.x_2,
                        y_2=block.y_2
                    ),
                    content=content,
                    confidence=block.layout_score
                )

                # Handle table data
                if block.table_structure:
                    b.table_data = self._convert_table_structure(block.table_structure)

                page_doc.blocks.append(b)
                page_doc.page_content += block.markdown_content + "\n"

        return page_doc

    def _map_label_to_type(self, label: str) -> str:
        """Map PDF processor label to BlockType"""
        mapping = {
            'Text': BlockType.PARAGRAPH.value,
            'Title': BlockType.TITLE.value,
            'Header': BlockType.HEADING_1.value,
            'Footer': BlockType.FOOTER.value,
            'Figure': BlockType.FIGURE.value,
            'Table': BlockType.TABLE.value,
            'Toc': BlockType.TOC.value,
            'Figure caption': BlockType.CAPTION.value,
            'Table caption': BlockType.CAPTION.value,
        }
        return mapping.get(label, BlockType.TEXT.value)

    def _convert_table_structure(self, table_structure) -> dict:
        """Convert table structure to JSON format"""
        cells = []
        for cell in table_structure:
            cells.append({
                "row": cell.row_nums[0] if cell.row_nums else 0,
                "col": cell.column_nums[0] if cell.column_nums else 0,
                "row_span": len(cell.row_nums) if cell.row_nums else 1,
                "col_span": len(cell.column_nums) if cell.column_nums else 1,
                "content": cell.content or "",
                "is_header": cell.column_header or cell.projected_row_header
            })

        max_row = max((c["row"] for c in cells), default=0) + 1
        max_col = max((c["col"] for c in cells), default=0) + 1

        return {
            "rows": max_row,
            "cols": max_col,
            "cells": cells
        }

    def _parse_word(self, doc: Document, file_path: Path) -> None:
        """Parse Word document using markitdown"""
        result = self.parser.parse(file_path)

        doc.document_info.title = result.title
        doc.document_info.total_pages = 1
        doc.full_text = result.text
        doc.full_markdown = result.markdown

        page = Page(page_num=1, width=612, height=792)
        page.page_content = result.text
        page.blocks = self._parse_markdown_to_blocks(result.markdown)
        page.width = max((b.bbox.x_2 for b in page.blocks), default=750) + 50
        page.height = max((b.bbox.y_2 for b in page.blocks), default=792) + 50

        doc.pages = [page]

    def _parse_excel(self, doc: Document, file_path: Path) -> None:
        """Parse Excel spreadsheet"""
        result = self.parser.parse(file_path)

        doc.document_info.title = result.title
        doc.full_text = result.text
        doc.full_markdown = result.markdown

        # Parse sheets
        pages = self._parse_sheets_to_pages(result.markdown)
        doc.pages = pages
        doc.document_info.total_pages = len(pages)

    def _parse_powerpoint(self, doc: Document, file_path: Path) -> None:
        """Parse PowerPoint presentation"""
        result = self.parser.parse(file_path)

        doc.document_info.title = result.title
        doc.full_text = result.text
        doc.full_markdown = result.markdown

        # Parse slides
        pages = self._parse_slides_to_pages(result.markdown)
        doc.pages = pages
        doc.document_info.total_pages = len(pages)

    def _parse_markdown(self, doc: Document, file_path: Path) -> None:
        """Parse Markdown file"""
        result = self.parser.parse(file_path)

        doc.document_info.title = result.title
        doc.document_info.total_pages = 1
        doc.full_text = result.text
        doc.full_markdown = result.markdown

        page = Page(page_num=1, width=800, height=1000)
        page.page_content = result.text
        page.blocks = self._parse_markdown_to_blocks(result.markdown)

        doc.pages = [page]

    def _parse_image(self, doc: Document, file_path: Path) -> None:
        """Parse image file"""
        result = self.parser.parse(file_path)
        img = Image.open(file_path)
        width, height = img.size

        doc.document_info.title = result.title or f"Image: {file_path.name}"
        doc.document_info.total_pages = 1
        doc.full_text = result.text
        doc.full_markdown = result.markdown

        page = Page(page_num=1, width=width, height=height, page_content=result.text)

        if result.text.strip():
            block = Block(
                block_id="p1_b001",
                type=BlockType.TEXT.value,
                bbox=BBox(x_1=0, y_1=0, x_2=width, y_2=height),
                content=Content(text=result.text, markdown=result.markdown),
                confidence=0.9
            )
            page.blocks = [block]

        doc.pages = [page]

        # Add image metadata
        doc.metadata.custom = {
            "image_width": width,
            "image_height": height,
            "image_format": result.metadata.get('format', ''),
            "image_mode": result.metadata.get('mode', '')
        }

    def _parse_generic(self, doc: Document, file_path: Path) -> None:
        """Generic parsing for unknown types"""
        result = self.parser.parse(file_path)

        doc.full_text = result.text
        doc.full_markdown = result.markdown

        page = Page(page_num=1, width=612, height=792)
        page.page_content = result.text
        doc.pages = [page]
        doc.document_info.total_pages = 1

    def _parse_markdown_to_blocks(self, markdown: str) -> List[Block]:
        """Parse markdown into structured blocks"""
        blocks = []
        lines = markdown.split('\n')
        block_id = 0
        y = 50

        patterns = {
            'h1': re.compile(r'^#\s+(.+)$'),
            'h2': re.compile(r'^##\s+(.+)$'),
            'h3': re.compile(r'^###\s+(.+)$'),
            'list': re.compile(r'^[\s]*[-*+]\s+(.+)$'),
            'table': re.compile(r'^\|.*\|$'),
        }

        for line in lines:
            block_id += 1

            # H1
            m = patterns['h1'].match(line)
            if m:
                blocks.append(Block(
                    block_id=f"p1_b{block_id:03d}",
                    type=BlockType.TITLE.value,
                    bbox=BBox(50, y, 750, y + 30),
                    content=Content(text=m.group(1), markdown=line),
                    properties={"font_size": 24, "font_weight": "bold"}
                ))
                y += 40
                continue

            # H2
            m = patterns['h2'].match(line)
            if m:
                blocks.append(Block(
                    block_id=f"p1_b{block_id:03d}",
                    type=BlockType.HEADING_1.value,
                    bbox=BBox(50, y, 750, y + 24),
                    content=Content(text=m.group(1), markdown=line),
                    properties={"font_size": 18, "font_weight": "bold"}
                ))
                y += 30
                continue

            # Skip table separator
            if re.match(r'^\|[\s\-:|]+\|$', line):
                continue

            # Table row
            if patterns['table'].match(line):
                cells = [c.strip() for c in line.split('|')[1:-1]]
                blocks.append(Block(
                    block_id=f"p1_b{block_id:03d}",
                    type=BlockType.TABLE.value,
                    bbox=BBox(50, y, 750, y + 20),
                    content=Content(text=' | '.join(cells), markdown=line),
                    table_data={
                        "rows": 1,
                        "cols": len(cells),
                        "cells": [
                            {"row": 0, "col": i, "content": c, "is_header": i == 0}
                            for i, c in enumerate(cells)
                        ]
                    }
                ))
                y += 25
                continue

            # List
            m = patterns['list'].match(line)
            if m:
                blocks.append(Block(
                    block_id=f"p1_b{block_id:03d}",
                    type=BlockType.LIST_ITEM.value,
                    bbox=BBox(70, y, 750, y + 16),
                    content=Content(text=m.group(1), markdown=line),
                    properties={"list_style": "bullet"}
                ))
                y += 20
                continue

            # Empty
            if not line.strip():
                y += 10
                continue

            # Paragraph
            blocks.append(Block(
                block_id=f"p1_b{block_id:03d}",
                type=BlockType.PARAGRAPH.value,
                bbox=BBox(50, y, 750, y + 16),
                content=Content(text=line, markdown=line),
                properties={"font_size": 12}
            ))
            y += 20

        return blocks

    def _parse_sheets_to_pages(self, markdown: str) -> List[Page]:
        """Parse Excel markdown into pages"""
        pages = []
        parts = re.split(r'^##\s+(.+)$', markdown, flags=re.MULTILINE)

        page_num = 0
        for i in range(0, len(parts) - 1, 2):
            sheet_name = parts[i + 1]
            content = parts[i + 2] if i + 2 < len(parts) else ""
            page_num += 1

            page = Page(
                page_num=page_num,
                width=800,
                height=1000,
                page_content=content
            )
            page.blocks = [
                Block(
                    block_id=f"p{page_num}_b001",
                    type=BlockType.TABLE.value,
                    bbox=BBox(50, 50, 750, 950),
                    content=Content(text=content.strip(), markdown=f"## {sheet_name}\n{content}")
                )
            ]
            pages.append(page)

        if not pages:
            pages.append(Page(
                page_num=1,
                width=800,
                height=1000,
                page_content=markdown,
                blocks=[Block(
                    block_id="p1_b001",
                    type=BlockType.TABLE.value,
                    bbox=BBox(50, 50, 750, 950),
                    content=Content(text=markdown, markdown=markdown)
                )]
            ))

        return pages

    def _parse_slides_to_pages(self, markdown: str) -> List[Page]:
        """Parse PowerPoint markdown into pages (slides)"""
        pages = []
        parts = re.split(r'(<!--\s*Slide number:\s*\d+\s*-->)', markdown, flags=re.IGNORECASE)

        slide_content = ""
        page_num = 0

        for part in parts:
            if re.match(r'<!--\s*Slide number:', part, re.IGNORECASE):
                if slide_content.strip():
                    page_num += 1
                    page = Page(page_num=page_num, width=960, height=720, page_content=slide_content.strip())
                    page.blocks = self._parse_markdown_to_blocks(slide_content.strip())
                    pages.append(page)
                slide_content = ""
            else:
                slide_content += part

        if slide_content.strip():
            page_num += 1
            page = Page(page_num=page_num, width=960, height=720, page_content=slide_content.strip())
            page.blocks = self._parse_markdown_to_blocks(slide_content.strip())
            pages.append(page)

        if not pages:
            pages.append(Page(
                page_num=1,
                width=960,
                height=720,
                page_content=markdown,
                blocks=self._parse_markdown_to_blocks(markdown)
            ))

        return pages


# Convenience function
def parse_document(file_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None, vlm_url: Optional[str] = None) -> Document:
    """
    Parse a document file into uniform JSON format.

    Args:
        file_path: Path to document
        output_path: Optional path to save JSON
        vlm_url: Optional VLM server URL

    Returns:
        Document object
    """
    parser = DocumentParser(vlm_url=vlm_url)
    return parser.parse(file_path)
