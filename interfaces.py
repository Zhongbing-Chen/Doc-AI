"""
Abstract base classes (interfaces) for all processing backends

These interfaces define the contract for all backend implementations,
enabling the strategy pattern for easy backend swapping.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
from PIL import Image
import numpy as np


class LayoutDetectorBackend(ABC):
    """
    Abstract interface for layout detection backends

    Layout detection identifies different regions in a document image
    (text, tables, figures, headers, footers, etc.)
    """

    @abstractmethod
    def detect(self, image: Image.Image, **kwargs) -> Any:
        """
        Detect layout blocks in an image

        Args:
            image: PIL Image to process
            **kwargs: Backend-specific parameters (conf, iou, etc.)

        Returns:
            Backend-specific layout result
            - YOLO: ultralytics Results object
            - VLM: List of ContentBlock objects
            - None: None
        """
        pass

    @abstractmethod
    def initialize(self, config: dict) -> None:
        """
        Initialize the backend with configuration

        Args:
            config: Configuration dictionary containing backend-specific settings
        """
        pass

    def cleanup(self) -> None:
        """
        Cleanup resources (optional)

        Override this method if backend needs to release resources
        """
        pass


class TableParserBackend(ABC):
    """
    Abstract interface for table parsing backends

    Table parsing extracts structured data from table regions
    """

    @abstractmethod
    def parse(self, image: Image.Image, bbox: Optional[List[float]] = None, **kwargs) -> Any:
        """
        Parse table structure from an image

        Args:
            image: PIL Image containing the table
            bbox: Optional bounding box to crop [x1, y1, x2, y2]
            **kwargs: Backend-specific parameters

        Returns:
            Table structure
            - Table Transformer: List[TableStructure]
            - VLM: HTML string or List[TableStructure]
            - None: None
        """
        pass

    @abstractmethod
    def initialize(self, config: dict) -> None:
        """
        Initialize the backend with configuration

        Args:
            config: Configuration dictionary
        """
        pass

    def cleanup(self) -> None:
        """Cleanup resources (optional)"""
        pass


class TextExtractorBackend(ABC):
    """
    Abstract interface for text extraction backends

    Text extraction extracts text content from document regions
    """

    @abstractmethod
    def extract(self, image: Image.Image, bbox: Optional[List[float]] = None, **kwargs) -> str:
        """
        Extract text from an image region

        Args:
            image: PIL Image to process
            bbox: Optional bounding box to crop
            **kwargs: Backend-specific parameters

        Returns:
            Extracted text string
        """
        pass

    @abstractmethod
    def is_scanned(self, pdf_page) -> bool:
        """
        Determine if a PDF page is scanned (no text layer)

        Args:
            pdf_page: fitz.Page object

        Returns:
            True if scanned, False otherwise
        """
        pass

    @abstractmethod
    def initialize(self, config: dict) -> None:
        """
        Initialize the backend with configuration

        Args:
            config: Configuration dictionary
        """
        pass

    def cleanup(self) -> None:
        """Cleanup resources (optional)"""
        pass


class OrientationDetectorBackend(ABC):
    """
    Abstract interface for orientation detection backends

    Orientation detection identifies if a document is rotated
    and corrects it
    """

    @abstractmethod
    def detect(self, image: Image.Image, pdf_page, **kwargs) -> Tuple[Image.Image, int]:
        """
        Detect and correct orientation

        Args:
            image: PIL Image to process
            pdf_page: fitz.Page object (for rotation metadata)
            **kwargs: Backend-specific parameters

        Returns:
            Tuple of (rotated_image, rotation_angle)
            - rotation_angle: 0, 90, 180, or 270 degrees
        """
        pass

    @abstractmethod
    def initialize(self, config: dict) -> None:
        """
        Initialize the backend with configuration

        Args:
            config: Configuration dictionary
        """
        pass

    def cleanup(self) -> None:
        """Cleanup resources (optional)"""
        pass


class SkewDetectorBackend(ABC):
    """
    Abstract interface for skew detection backends

    Skew detection identifies and corrects small angle rotations
    in scanned documents
    """

    @abstractmethod
    def detect_and_correct(self, image: Image.Image, **kwargs) -> Tuple[Image.Image, float]:
        """
        Detect and correct skew

        Args:
            image: PIL Image to process
            **kwargs: Backend-specific parameters

        Returns:
            Tuple of (deskewed_image, skew_angle)
        """
        pass

    @abstractmethod
    def initialize(self, config: dict) -> None:
        """
        Initialize the backend with configuration

        Args:
            config: Configuration dictionary
        """
        pass

    def cleanup(self) -> None:
        """Cleanup resources (optional)"""
        pass