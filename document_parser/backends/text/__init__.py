"""
Text extraction backends
"""

from .fitz_backend import FitzTextBackend
from .rapidocr_backend import RapidOCRBackend
from .tesseract_backend import TesseractTextBackend
from .vlm_backend import VLMTextBackend
from .auto_backend import AutoTextBackend
from .none_backend import NoneTextBackend

__all__ = [
    'FitzTextBackend',
    'RapidOCRBackend',
    'TesseractTextBackend',
    'VLMTextBackend',
    'AutoTextBackend',
    'NoneTextBackend'
]