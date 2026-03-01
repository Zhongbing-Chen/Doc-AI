"""
Orientation detection backends
"""

from .tesseract_backend import TesseractOrientationBackend
from .vlm_backend import VLMOrientationBackend
from .none_backend import NoneOrientationBackend

__all__ = [
    'TesseractOrientationBackend',
    'VLMOrientationBackend',
    'NoneOrientationBackend'
]