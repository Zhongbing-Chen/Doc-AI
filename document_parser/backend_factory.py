"""
Factory for creating backend instances

This module provides a centralized factory for creating and managing
backend instances for each processing step.
"""
from typing import Any, Dict, Type
from .interfaces import (
    LayoutDetectorBackend,
    TableParserBackend,
    TextExtractorBackend,
    OrientationDetectorBackend,
    SkewDetectorBackend
)


class BackendFactory:
    """
    Factory for creating processing backends

    This factory uses a registry pattern to manage backend creation.
    Backends are registered by name and created on demand.
    """

    # Backend registries
    _layout_backends: Dict[str, Type[LayoutDetectorBackend]] = {}
    _table_backends: Dict[str, Type[TableParserBackend]] = {}
    _text_backends: Dict[str, Type[TextExtractorBackend]] = {}
    _orientation_backends: Dict[str, Type[OrientationDetectorBackend]] = {}
    _skew_backends: Dict[str, Type[SkewDetectorBackend]] = {}

    @classmethod
    def register_layout_backend(cls, name: str, backend_class: Type[LayoutDetectorBackend]) -> None:
        """
        Register a layout detection backend

        Args:
            name: Backend name (e.g., 'yolo', 'vlm', 'none')
            backend_class: Backend class (must inherit from LayoutDetectorBackend)
        """
        cls._layout_backends[name] = backend_class

    @classmethod
    def register_table_backend(cls, name: str, backend_class: Type[TableParserBackend]) -> None:
        """
        Register a table parsing backend

        Args:
            name: Backend name
            backend_class: Backend class
        """
        cls._table_backends[name] = backend_class

    @classmethod
    def register_text_backend(cls, name: str, backend_class: Type[TextExtractorBackend]) -> None:
        """
        Register a text extraction backend

        Args:
            name: Backend name
            backend_class: Backend class
        """
        cls._text_backends[name] = backend_class

    @classmethod
    def register_orientation_backend(cls, name: str, backend_class: Type[OrientationDetectorBackend]) -> None:
        """
        Register an orientation detection backend

        Args:
            name: Backend name
            backend_class: Backend class
        """
        cls._orientation_backends[name] = backend_class

    @classmethod
    def register_skew_backend(cls, name: str, backend_class: Type[SkewDetectorBackend]) -> None:
        """
        Register a skew detection backend

        Args:
            name: Backend name
            backend_class: Backend class
        """
        cls._skew_backends[name] = backend_class

    @classmethod
    def create_layout_backend(cls, backend_type: str, config: Dict[str, Any]) -> LayoutDetectorBackend:
        """
        Create a layout detection backend

        Args:
            backend_type: Backend name (e.g., 'yolo', 'vlm', 'none')
            config: Configuration dictionary

        Returns:
            Initialized backend instance

        Raises:
            ValueError: If backend type is not registered
        """
        if backend_type not in cls._layout_backends:
            raise ValueError(
                f"Unknown layout backend: '{backend_type}'. "
                f"Available: {list(cls._layout_backends.keys())}"
            )

        backend = cls._layout_backends[backend_type]()
        backend.initialize(config)
        return backend

    @classmethod
    def create_table_backend(cls, backend_type: str, config: Dict[str, Any]) -> TableParserBackend:
        """
        Create a table parsing backend

        Args:
            backend_type: Backend name
            config: Configuration dictionary

        Returns:
            Initialized backend instance

        Raises:
            ValueError: If backend type is not registered
        """
        if backend_type not in cls._table_backends:
            raise ValueError(
                f"Unknown table backend: '{backend_type}'. "
                f"Available: {list(cls._table_backends.keys())}"
            )

        backend = cls._table_backends[backend_type]()
        backend.initialize(config)
        return backend

    @classmethod
    def create_text_backend(cls, backend_type: str, config: Dict[str, Any]) -> TextExtractorBackend:
        """
        Create a text extraction backend

        Args:
            backend_type: Backend name
            config: Configuration dictionary

        Returns:
            Initialized backend instance

        Raises:
            ValueError: If backend type is not registered
        """
        if backend_type not in cls._text_backends:
            raise ValueError(
                f"Unknown text backend: '{backend_type}'. "
                f"Available: {list(cls._text_backends.keys())}"
            )

        backend = cls._text_backends[backend_type]()
        backend.initialize(config)
        return backend

    @classmethod
    def create_orientation_backend(cls, backend_type: str, config: Dict[str, Any]) -> OrientationDetectorBackend:
        """
        Create an orientation detection backend

        Args:
            backend_type: Backend name
            config: Configuration dictionary

        Returns:
            Initialized backend instance

        Raises:
            ValueError: If backend type is not registered
        """
        if backend_type not in cls._orientation_backends:
            raise ValueError(
                f"Unknown orientation backend: '{backend_type}'. "
                f"Available: {list(cls._orientation_backends.keys())}"
            )

        backend = cls._orientation_backends[backend_type]()
        backend.initialize(config)
        return backend

    @classmethod
    def create_skew_backend(cls, backend_type: str, config: Dict[str, Any]) -> SkewDetectorBackend:
        """
        Create a skew detection backend

        Args:
            backend_type: Backend name
            config: Configuration dictionary

        Returns:
            Initialized backend instance

        Raises:
            ValueError: If backend type is not registered
        """
        if backend_type not in cls._skew_backends:
            raise ValueError(
                f"Unknown skew backend: '{backend_type}'. "
                f"Available: {list(cls._skew_backends.keys())}"
            )

        backend = cls._skew_backends[backend_type]()
        backend.initialize(config)
        return backend

    @classmethod
    def list_layout_backends(cls) -> list:
        """List all registered layout backends"""
        return list(cls._layout_backends.keys())

    @classmethod
    def list_table_backends(cls) -> list:
        """List all registered table backends"""
        return list(cls._table_backends.keys())

    @classmethod
    def list_text_backends(cls) -> list:
        """List all registered text backends"""
        return list(cls._text_backends.keys())

    @classmethod
    def list_orientation_backends(cls) -> list:
        """List all registered orientation backends"""
        return list(cls._orientation_backends.keys())

    @classmethod
    def list_skew_backends(cls) -> list:
        """List all registered skew backends"""
        return list(cls._skew_backends.keys())


def register_all_backends():
    """
    Register all default backends

    This function is called automatically when the module is imported.
    It registers all built-in backends with the factory.
    """
    # Import backends here to avoid circular imports
    # Layout backends
    try:
        from .backends.layout import YOLOLayoutBackend, VLMLayoutBackend, NoneLayoutBackend
        BackendFactory.register_layout_backend('yolo', YOLOLayoutBackend)
        BackendFactory.register_layout_backend('vlm', VLMLayoutBackend)
        BackendFactory.register_layout_backend('none', NoneLayoutBackend)
    except ImportError as e:
        print(f"Warning: Could not import layout backends: {e}")

    # Table backends
    try:
        from .backends.table import TableTransformerBackend, VLMTableBackend, NoneTableBackend
        BackendFactory.register_table_backend('table_transformer', TableTransformerBackend)
        BackendFactory.register_table_backend('vlm', VLMTableBackend)
        BackendFactory.register_table_backend('none', NoneTableBackend)
    except ImportError as e:
        print(f"Warning: Could not import table backends: {e}")

    # Text backends
    try:
        from .backends.text import (
            FitzTextBackend,
            RapidOCRBackend,
            TesseractTextBackend,
            VLMTextBackend,
            AutoTextBackend,
            NoneTextBackend
        )
        BackendFactory.register_text_backend('fitz', FitzTextBackend)
        BackendFactory.register_text_backend('rapidocr', RapidOCRBackend)
        BackendFactory.register_text_backend('tesseract', TesseractTextBackend)
        BackendFactory.register_text_backend('vlm', VLMTextBackend)
        BackendFactory.register_text_backend('auto', AutoTextBackend)
        BackendFactory.register_text_backend('none', NoneTextBackend)
    except ImportError as e:
        print(f"Warning: Could not import text backends: {e}")

    # Orientation backends
    try:
        from .backends.orientation import TesseractOrientationBackend, VLMOrientationBackend, NoneOrientationBackend
        BackendFactory.register_orientation_backend('tesseract', TesseractOrientationBackend)
        BackendFactory.register_orientation_backend('vlm', VLMOrientationBackend)
        BackendFactory.register_orientation_backend('none', NoneOrientationBackend)
    except ImportError as e:
        print(f"Warning: Could not import orientation backends: {e}")

    # Skew backends
    try:
        from .backends.skew import JdeskewBackend, NoneSkewBackend
        BackendFactory.register_skew_backend('jdeskew', JdeskewBackend)
        BackendFactory.register_skew_backend('none', NoneSkewBackend)
    except ImportError as e:
        print(f"Warning: Could not import skew backends: {e}")


# Auto-register backends when module is imported
register_all_backends()