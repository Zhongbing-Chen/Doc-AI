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

__all__ = [
    'ParserConfig',
    'LayoutConfig',
    'TableConfig',
    'TextConfig',
    'OrientationConfig',
    'SkewConfig',
    'VLMConfig',
    'ConfigurablePDFProcessor',
]