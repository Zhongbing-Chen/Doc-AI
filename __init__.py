"""
Document Parser - Configurable document parsing with pluggable backends

This module provides a fully configurable document parsing system where each
processing step can use different backends (YOLO, VLM, OCR, etc.)
"""

from .config import (
    ParserConfig,
    LayoutConfig,
    TableConfig,
    TextConfig,
    OrientationConfig,
    SkewConfig,
    VLMConfig
)
from .processor import ConfigurablePDFProcessor
from .universal_parser import (
    UniversalDocumentParser,
    ParseResult,
    parse_document
)
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
    DocumentType
)
from .parser import (
    DocumentParser,
    parse_document as parse_to_json
)

__all__ = [
    'ParserConfig',
    'LayoutConfig',
    'TableConfig',
    'TextConfig',
    'OrientationConfig',
    'SkewConfig',
    'VLMConfig',
    'ConfigurablePDFProcessor',
    'UniversalDocumentParser',
    'ParseResult',
    'parse_document',
    'Document',
    'DocumentMetadata',
    'DocumentInfo',
    'Page',
    'Block',
    'BBox',
    'Content',
    'TableCell',
    'BlockType',
    'DocumentType',
    'DocumentParser',
    'parse_to_json',
]