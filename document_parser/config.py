"""
Configuration classes for document parsing - Parameter-based configuration
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class LayoutBackend(Enum):
    """Available backends for layout detection"""
    NONE = "none"
    YOLO = "yolo"
    VLM = "vlm"


class TableBackend(Enum):
    """Available backends for table parsing"""
    NONE = "none"
    TABLE_TRANSFORMER = "table_transformer"
    VLM = "vlm"


class TextBackend(Enum):
    """Available backends for text extraction"""
    NONE = "none"
    FITZ = "fitz"
    RAPIDOCR = "rapidocr"
    TESSERACT = "tesseract"
    VLM = "vlm"
    AUTO = "auto"


class OrientationBackend(Enum):
    """Available backends for orientation detection"""
    NONE = "none"
    TESSERACT = "tesseract"
    VLM = "vlm"


class SkewBackend(Enum):
    """Available backends for skew detection"""
    NONE = "none"
    JDESKEW = "jdeskew"


@dataclass
class VLMConfig:
    """Configuration for VLM (Vision Language Model) backends"""
    backend: str = "http-client"
    server_url: str = "http://localhost:8000"
    server_headers: Optional[Dict[str, str]] = None
    max_concurrency: int = 100
    http_timeout: int = 600
    use_tqdm: bool = True
    debug: bool = False
    max_retries: int = 3
    retry_backoff_factor: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'backend': self.backend,
            'server_url': self.server_url,
            'server_headers': self.server_headers,
            'max_concurrency': self.max_concurrency,
            'http_timeout': self.http_timeout,
            'use_tqdm': self.use_tqdm,
            'debug': self.debug,
            'max_retries': self.max_retries,
            'retry_backoff_factor': self.retry_backoff_factor
        }


@dataclass
class LayoutConfig:
    """Configuration for layout detection"""
    enabled: bool = True
    backend: str = LayoutBackend.YOLO.value
    confidence: float = 0.5
    iou_threshold: float = 0.45
    model_source: str = "huggingface"  # huggingface, modelscope, or local path
    model_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'enabled': self.enabled,
            'backend': self.backend,
            'confidence': self.confidence,
            'iou_threshold': self.iou_threshold,
            'model_source': self.model_source,
            'model_path': self.model_path
        }


@dataclass
class TableConfig:
    """Configuration for table parsing"""
    enabled: bool = True
    backend: str = TableBackend.TABLE_TRANSFORMER.value
    model_source: str = "huggingface"
    model_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'enabled': self.enabled,
            'backend': self.backend,
            'model_source': self.model_source,
            'model_path': self.model_path
        }


@dataclass
class TextConfig:
    """Configuration for text extraction"""
    enabled: bool = True
    backend: str = TextBackend.AUTO.value
    ocr_engine: str = "rapidocr"  # rapidocr or tesseract
    languages: List[str] = field(default_factory=lambda: ["chi_sim", "eng"])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'enabled': self.enabled,
            'backend': self.backend,
            'ocr_engine': self.ocr_engine,
            'languages': self.languages
        }


@dataclass
class OrientationConfig:
    """Configuration for orientation detection"""
    enabled: bool = True
    backend: str = OrientationBackend.TESSERACT.value
    confidence_threshold: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'enabled': self.enabled,
            'backend': self.backend,
            'confidence_threshold': self.confidence_threshold
        }


@dataclass
class SkewConfig:
    """Configuration for skew detection"""
    enabled: bool = True
    backend: str = SkewBackend.JDESKEW.value
    angle_max: float = 10.0
    threshold: float = 0.3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'enabled': self.enabled,
            'backend': self.backend,
            'angle_max': self.angle_max,
            'threshold': self.threshold
        }


@dataclass
class ParserConfig:
    """
    Main configuration for document parser

    All parameters have sensible defaults matching current PDFProcessor behavior.

    Example:
        # Simple usage with defaults
        config = ParserConfig(device="cuda")

        # Custom layout backend
        config = ParserConfig(
            device="cuda",
            layout=LayoutConfig(backend=LayoutBackend.VLM.value)
        )

        # Full VLM pipeline
        config = ParserConfig(
            device="cuda",
            layout=LayoutConfig(backend=LayoutBackend.VLM.value),
            table=TableConfig(backend=TableBackend.VLM.value),
            text=TextConfig(backend=TextBackend.VLM.value),
            vlm=VLMConfig(server_url="http://localhost:8000")
        )
    """
    # Global settings
    device: str = "cpu"
    zoom_factor: float = 3.0
    output_dir: str = "./results"
    visualization: bool = False

    # Processing step configurations
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    table: TableConfig = field(default_factory=TableConfig)
    text: TextConfig = field(default_factory=TextConfig)
    orientation: OrientationConfig = field(default_factory=OrientationConfig)
    skew: SkewConfig = field(default_factory=SkewConfig)

    # VLM configuration (shared across all VLM backends)
    vlm: VLMConfig = field(default_factory=VLMConfig)

    # Advanced settings
    multiprocessing_enabled: bool = False
    max_workers: Optional[int] = None

    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()

    def validate(self) -> bool:
        """
        Validate configuration

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate device
        if self.device not in ['cpu', 'cuda', 'mps']:
            raise ValueError(f"Invalid device: {self.device}. Must be 'cpu', 'cuda', or 'mps'")

        # Validate zoom_factor
        if self.zoom_factor <= 0:
            raise ValueError(f"Invalid zoom_factor: {self.zoom_factor}. Must be positive")

        # Validate backends
        valid_layout = [b.value for b in LayoutBackend]
        if self.layout.backend not in valid_layout:
            raise ValueError(f"Invalid layout backend: {self.layout.backend}. Valid options: {valid_layout}")

        valid_table = [b.value for b in TableBackend]
        if self.table.backend not in valid_table:
            raise ValueError(f"Invalid table backend: {self.table.backend}. Valid options: {valid_table}")

        valid_text = [b.value for b in TextBackend]
        if self.text.backend not in valid_text:
            raise ValueError(f"Invalid text backend: {self.text.backend}. Valid options: {valid_text}")

        valid_orientation = [b.value for b in OrientationBackend]
        if self.orientation.backend not in valid_orientation:
            raise ValueError(f"Invalid orientation backend: {self.orientation.backend}. Valid options: {valid_orientation}")

        valid_skew = [b.value for b in SkewBackend]
        if self.skew.backend not in valid_skew:
            raise ValueError(f"Invalid skew backend: {self.skew.backend}. Valid options: {valid_skew}")

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'device': self.device,
            'zoom_factor': self.zoom_factor,
            'output_dir': self.output_dir,
            'visualization': self.visualization,
            'layout': self.layout.to_dict(),
            'table': self.table.to_dict(),
            'text': self.text.to_dict(),
            'orientation': self.orientation.to_dict(),
            'skew': self.skew.to_dict(),
            'vlm': self.vlm.to_dict(),
            'multiprocessing_enabled': self.multiprocessing_enabled,
            'max_workers': self.max_workers
        }

    @classmethod
    def default(cls) -> 'ParserConfig':
        """
        Get default configuration (backward compatible)

        Returns:
            ParserConfig with default settings
        """
        return cls()

    @classmethod
    def fast(cls) -> 'ParserConfig':
        """
        Fast configuration (YOLO for layout, no table, fitz for text)

        Returns:
            ParserConfig optimized for speed
        """
        return cls(
            layout=LayoutConfig(backend=LayoutBackend.YOLO.value, confidence=0.5),
            table=TableConfig(enabled=False),
            text=TextConfig(backend=TextBackend.FITZ.value),
            orientation=OrientationConfig(enabled=False),
            skew=SkewConfig(enabled=False)
        )

    @classmethod
    def accurate(cls, vlm_server_url: str = "http://localhost:8000") -> 'ParserConfig':
        """
        Accurate configuration (VLM for all steps)

        Args:
            vlm_server_url: URL of VLM server

        Returns:
            ParserConfig optimized for accuracy
        """
        return cls(
            layout=LayoutConfig(backend=LayoutBackend.VLM.value),
            table=TableConfig(backend=TableBackend.VLM.value),
            text=TextConfig(backend=TextBackend.VLM.value),
            orientation=OrientationConfig(backend=OrientationBackend.VLM.value),
            skew=SkewConfig(backend=SkewBackend.JDESKEW.value),
            vlm=VLMConfig(server_url=vlm_server_url)
        )

    @classmethod
    def scanned(cls, vlm_server_url: str = "http://localhost:8000") -> 'ParserConfig':
        """
        Configuration optimized for scanned documents

        Args:
            vlm_server_url: URL of VLM server

        Returns:
            ParserConfig optimized for scanned documents
        """
        return cls(
            orientation=OrientationConfig(backend=OrientationBackend.TESSERACT.value, enabled=True),
            skew=SkewConfig(backend=SkewBackend.JDESKEW.value, enabled=True),
            layout=LayoutConfig(backend=LayoutBackend.VLM.value),
            table=TableConfig(backend=TableBackend.VLM.value),
            text=TextConfig(backend=TextBackend.VLM.value),
            vlm=VLMConfig(server_url=vlm_server_url)
        )