"""
Universal Document Parser

This module provides a unified interface for parsing various document formats:
- Word documents (.doc, .docx)
- Excel spreadsheets (.xlsx, .xls)
- PowerPoint presentations (.ppt, .pptx)
- Markdown files (.md)
- Images (.png, .jpg, .jpeg)

Uses MarkItDown as the primary parsing engine for Office documents.
"""

import os
from pathlib import Path
from typing import Optional, Union
from PIL import Image

from markitdown import MarkItDown


class UniversalDocumentParser:
    """
    Universal document parser for multiple file formats

    This class provides a unified interface for parsing various document
    formats and converting them to markdown or structured text.
    """

    # Supported file extensions
    SUPPORTED_FORMATS = {
        '.doc': 'word',
        '.docx': 'word',
        '.xls': 'excel',
        '.xlsx': 'excel',
        '.ppt': 'powerpoint',
        '.pptx': 'powerpoint',
        '.md': 'markdown',
        '.markdown': 'markdown',
        '.png': 'image',
        '.jpg': 'image',
        '.jpeg': 'image',
    }

    def __init__(self,
                 enable_vlm: bool = False,
                 vlm_url: Optional[str] = None,
                 **kwargs):
        """
        Initialize the universal document parser

        Args:
            enable_vlm: Enable VLM for image parsing (default: False)
            vlm_url: URL for VLM server (optional, for advanced image parsing)
            **kwargs: Additional arguments passed to MarkItDown
        """
        self.enable_vlm = enable_vlm
        self.vlm_url = vlm_url
        self.markitdown = MarkItDown(**kwargs)

    def parse(self, file_path: Union[str, Path]) -> 'ParseResult':
        """
        Parse a document file

        Args:
            file_path: Path to the document file

        Returns:
            ParseResult object containing parsed content

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = file_path.suffix.lower()

        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                f"Supported formats: {list(self.SUPPORTED_FORMATS.keys())}"
            )

        file_type = self.SUPPORTED_FORMATS[file_ext]

        # Parse based on file type
        if file_type == 'markdown':
            result = self._parse_markdown(file_path)
        elif file_type == 'image':
            result = self._parse_image(file_path)
        else:
            # Use MarkItDown for Office documents
            result = self._parse_with_markitdown(file_path, file_type)

        result.file_path = str(file_path)
        result.file_type = file_type
        result.file_ext = file_ext

        return result

    def _parse_with_markitdown(self, file_path: Path, file_type: str) -> 'ParseResult':
        """
        Parse Office documents using MarkItDown

        Args:
            file_path: Path to the document
            file_type: Type of document (word, excel, powerpoint)

        Returns:
            ParseResult object
        """
        result = self.markitdown.convert(file_path)

        return ParseResult(
            text=result.text_content,
            markdown=result.text_content,
            title=self._extract_title(result.text_content, file_type),
            metadata={
                'file_type': file_type,
                'source': 'markitdown'
            }
        )

    def _parse_markdown(self, file_path: Path) -> 'ParseResult':
        """
        Parse markdown file

        Args:
            file_path: Path to markdown file

        Returns:
            ParseResult object
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return ParseResult(
            text=content,
            markdown=content,
            title=self._extract_markdown_title(content),
            metadata={
                'file_type': 'markdown',
                'source': 'direct_read'
            }
        )

    def _parse_image(self, file_path: Path) -> 'ParseResult':
        """
        Parse image file

        For images, MarkItDown can extract embedded text or describe the image
        if VLM is enabled.

        Args:
            file_path: Path to image file

        Returns:
            ParseResult object
        """
        # Use MarkItDown for image parsing
        result = self.markitdown.convert(file_path)

        # Get image metadata
        img = Image.open(file_path)
        metadata = {
            'file_type': 'image',
            'source': 'markitdown',
            'width': img.width,
            'height': img.height,
            'format': img.format,
            'mode': img.mode
        }

        return ParseResult(
            text=result.text_content,
            markdown=result.text_content,
            title=f"Image: {file_path.name}",
            metadata=metadata
        )

    def _extract_title(self, content: str, file_type: str) -> str:
        """
        Extract title from document content

        Args:
            content: Document content
            file_type: Type of document

        Returns:
            Extracted title or empty string
        """
        if not content:
            return ""

        # For Word documents, try to find first heading
        if file_type == 'word':
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('# '):
                    return line[2:].strip()
                elif line and not line.startswith('#'):
                    # Return first non-empty line if no heading found
                    return line[:100]

        return ""

    def _extract_markdown_title(self, content: str) -> str:
        """
        Extract title from markdown content

        Args:
            content: Markdown content

        Returns:
            Extracted title or empty string
        """
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()

        return ""

    def parse_to_markdown(self, file_path: Union[str, Path]) -> str:
        """
        Parse document and return markdown string

        Args:
            file_path: Path to document file

        Returns:
            Markdown content string
        """
        result = self.parse(file_path)
        return result.markdown

    def get_supported_formats(self) -> dict:
        """
        Get dictionary of supported file formats

        Returns:
            Dictionary mapping extensions to file types
        """
        return self.SUPPORTED_FORMATS.copy()


class ParseResult:
    """
    Result of parsing a document

    Attributes:
        text: Plain text content
        markdown: Markdown formatted content
        title: Document title (if available)
        metadata: Dictionary of metadata about the document
        file_path: Path to the parsed file
        file_type: Type of document (word, excel, powerpoint, markdown, image)
        file_ext: File extension
    """

    def __init__(self,
                 text: str = "",
                 markdown: str = "",
                 title: str = "",
                 metadata: Optional[dict] = None):
        self.text = text
        self.markdown = markdown
        self.title = title
        self.metadata = metadata or {}
        self.file_path: str = ""
        self.file_type: str = ""
        self.file_ext: str = ""

    def __repr__(self) -> str:
        return (f"ParseResult(file_type='{self.file_type}', "
                f"file_ext='{self.file_ext}', "
                f"text_length={len(self.text)})")

    def to_dict(self) -> dict:
        """
        Convert result to dictionary

        Returns:
            Dictionary with all result data
        """
        return {
            'file_path': self.file_path,
            'file_type': self.file_type,
            'file_ext': self.file_ext,
            'text': self.text,
            'markdown': self.markdown,
            'title': self.title,
            'metadata': self.metadata
        }


# Convenience function for quick parsing
def parse_document(file_path: Union[str, Path], **kwargs) -> ParseResult:
    """
    Quick function to parse any supported document

    Args:
        file_path: Path to document file
        **kwargs: Arguments passed to UniversalDocumentParser

    Returns:
        ParseResult object
    """
    parser = UniversalDocumentParser(**kwargs)
    return parser.parse(file_path)
