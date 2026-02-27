"""
Configuration loader with validation and default handling for document parsing
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ProcessingStepConfig:
    """Configuration for a single processing step"""
    enabled: bool = True
    backend: str = "none"
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsingConfig:
    """
    Complete parsing configuration

    Supports loading from:
    - YAML files
    - Python dictionaries
    - Default configuration (backward compatible)
    """
    # Global settings
    device: str = "cpu"
    zoom_factor: float = 3.0
    output_dir: str = "./results"
    visualization: bool = False

    # Pipeline steps
    orientation: ProcessingStepConfig = field(default_factory=lambda: ProcessingStepConfig(enabled=True, backend="tesseract"))
    skew: ProcessingStepConfig = field(default_factory=lambda: ProcessingStepConfig(enabled=True, backend="jdeskew"))
    layout: ProcessingStepConfig = field(default_factory=lambda: ProcessingStepConfig(enabled=True, backend="yolo"))
    table: ProcessingStepConfig = field(default_factory=lambda: ProcessingStepConfig(enabled=True, backend="table_transformer"))
    text: ProcessingStepConfig = field(default_factory=lambda: ProcessingStepConfig(enabled=True, backend="auto"))

    # VLM settings
    vlm_config: Dict[str, Any] = field(default_factory=lambda: {
        "backend": "http-client",
        "server_url": "http://localhost:8000",
        "server_headers": None,
        "max_concurrency": 100,
        "http_timeout": 600,
        "use_tqdm": True,
        "debug": False,
        "max_retries": 3,
        "retry_backoff_factor": 0.5
    })

    # Advanced settings
    multiprocessing_enabled: bool = False
    max_workers: Optional[int] = None

    @classmethod
    def from_yaml(cls, config_path: str) -> 'ParsingConfig':
        """
        Load configuration from YAML file

        Args:
            config_path: Path to YAML configuration file

        Returns:
            ParsingConfig object

        Example:
            >>> config = ParsingConfig.from_yaml("configs/parsing_config.yaml")
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParsingConfig':
        """
        Create configuration from dictionary

        Args:
            data: Configuration dictionary

        Returns:
            ParsingConfig object
        """
        if not data:
            return cls.default()

        # Extract global settings
        global_cfg = data.get('global', {})

        # Extract pipeline configuration
        pipeline = data.get('pipeline', {})

        # Parse each processing step
        orientation = cls._parse_step(pipeline.get('orientation', {}))
        skew = cls._parse_step(pipeline.get('skew', {}))
        layout = cls._parse_step(pipeline.get('layout', {}))
        table = cls._parse_step(pipeline.get('table', {}))
        text = cls._parse_step(pipeline.get('text', {}))

        # Extract VLM config
        vlm_cfg = data.get('vlm', {})

        # Extract advanced settings
        advanced = data.get('advanced', {})

        return cls(
            device=global_cfg.get('device', 'cpu'),
            zoom_factor=global_cfg.get('zoom_factor', 3.0),
            output_dir=global_cfg.get('output_dir', './results'),
            visualization=global_cfg.get('visualization', False),
            orientation=orientation,
            skew=skew,
            layout=layout,
            table=table,
            text=text,
            vlm_config=vlm_cfg if vlm_cfg else cls.__dataclass_fields__['vlm_config'].default_factory(),
            multiprocessing_enabled=advanced.get('multiprocessing', {}).get('enabled', False),
            max_workers=advanced.get('multiprocessing', {}).get('max_workers')
        )

    @classmethod
    def default(cls) -> 'ParsingConfig':
        """
        Create default configuration (backward compatible with existing PDFProcessor)

        Returns:
            ParsingConfig with default settings matching current behavior

        Note:
            Default configuration uses:
            - YOLO for layout detection
            - Table Transformer for table parsing
            - Auto selection for text extraction
            - Tesseract for orientation
            - Jdeskew for skew correction
        """
        return cls()

    @staticmethod
    def _parse_step(step_data: Dict[str, Any]) -> ProcessingStepConfig:
        """
        Parse a single processing step configuration

        Args:
            step_data: Step configuration dictionary

        Returns:
            ProcessingStepConfig object
        """
        if not step_data:
            return ProcessingStepConfig()

        return ProcessingStepConfig(
            enabled=step_data.get('enabled', True),
            backend=step_data.get('backend', 'none'),
            config=step_data.get('config', {})
        )

    def validate(self) -> bool:
        """
        Validate configuration

        Returns:
            True if configuration is valid

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
        valid_layout_backends = ['none', 'yolo', 'vlm']
        if self.layout.backend not in valid_layout_backends:
            raise ValueError(f"Invalid layout backend: {self.layout.backend}. Must be one of {valid_layout_backends}")

        valid_table_backends = ['none', 'table_transformer', 'vlm']
        if self.table.backend not in valid_table_backends:
            raise ValueError(f"Invalid table backend: {self.table.backend}. Must be one of {valid_table_backends}")

        valid_text_backends = ['none', 'fitz', 'rapidocr', 'tesseract', 'vlm', 'auto']
        if self.text.backend not in valid_text_backends:
            raise ValueError(f"Invalid text backend: {self.text.backend}. Must be one of {valid_text_backends}")

        valid_orientation_backends = ['none', 'tesseract', 'vlm']
        if self.orientation.backend not in valid_orientation_backends:
            raise ValueError(f"Invalid orientation backend: {self.orientation.backend}. Must be one of {valid_orientation_backends}")

        valid_skew_backends = ['none', 'jdeskew']
        if self.skew.backend not in valid_skew_backends:
            raise ValueError(f"Invalid skew backend: {self.skew.backend}. Must be one of {valid_skew_backends}")

        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary

        Returns:
            Configuration as dictionary
        """
        return {
            'global': {
                'device': self.device,
                'zoom_factor': self.zoom_factor,
                'output_dir': self.output_dir,
                'visualization': self.visualization
            },
            'pipeline': {
                'orientation': {
                    'enabled': self.orientation.enabled,
                    'backend': self.orientation.backend,
                    'config': self.orientation.config
                },
                'skew': {
                    'enabled': self.skew.enabled,
                    'backend': self.skew.backend,
                    'config': self.skew.config
                },
                'layout': {
                    'enabled': self.layout.enabled,
                    'backend': self.layout.backend,
                    'config': self.layout.config
                },
                'table': {
                    'enabled': self.table.enabled,
                    'backend': self.table.backend,
                    'config': self.table.config
                },
                'text': {
                    'enabled': self.text.enabled,
                    'backend': self.text.backend,
                    'config': self.text.config
                }
            },
            'vlm': self.vlm_config,
            'advanced': {
                'multiprocessing': {
                    'enabled': self.multiprocessing_enabled,
                    'max_workers': self.max_workers
                }
            }
        }

    def save_yaml(self, filepath: str) -> None:
        """
        Save configuration to YAML file

        Args:
            filepath: Path to save YAML file
        """
        data = self.to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)