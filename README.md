# Document Parser - Configurable Document Parsing System

A fully configurable document parsing system where each processing step can use different backends (YOLO, VLM, OCR, etc.).

## Quick Start

### Basic Usage (Backward Compatible)

```python
from process import PDFProcessor

# Use existing API - works exactly as before
processor = PDFProcessor(device="cpu", zoom_factor=3)
pages = processor.process("document.pdf")
```

### New Configurable API

```python
from document_parser import ParserConfig, ConfigurablePDFProcessor

# Create configuration
config = ParserConfig(device="cuda")

# Create processor with config
processor = ConfigurablePDFProcessor(config)
pages = processor.process("document.pdf")
```

## Configuration Options

### Global Settings

```python
from document_parser import ParserConfig

config = ParserConfig(
    device="cuda",              # cpu, cuda, mps
    zoom_factor=3.0,            # Image zoom factor
    output_dir="./results",     # Output directory
    visualization=False         # Enable visualization
)
```

### Processing Step Configuration

Each processing step can be configured independently:

```python
from document_parser import (
    ParserConfig,
    LayoutConfig,
    TableConfig,
    TextConfig,
    OrientationConfig,
    SkewConfig,
    VLMConfig
)

config = ParserConfig(
    # Layout detection
    layout=LayoutConfig(
        enabled=True,
        backend="yolo",         # Options: "none", "yolo", "vlm"
        confidence=0.5,
        iou_threshold=0.45,
        model_source="huggingface"
    ),

    # Table parsing
    table=TableConfig(
        enabled=True,
        backend="table_transformer",  # Options: "none", "table_transformer", "vlm"
        model_source="huggingface"
    ),

    # Text extraction
    text=TextConfig(
        enabled=True,
        backend="auto",         # Options: "none", "fitz", "rapidocr", "tesseract", "vlm", "auto"
        ocr_engine="rapidocr"   # Used when backend is "rapidocr" or "tesseract"
    ),

    # Orientation detection
    orientation=OrientationConfig(
        enabled=True,
        backend="tesseract",    # Options: "none", "tesseract", "vlm"
        confidence_threshold=0.5
    ),

    # Skew detection
    skew=SkewConfig(
        enabled=True,
        backend="jdeskew",      # Options: "none", "jdeskew"
        angle_max=10.0,
        threshold=0.3
    ),

    # VLM configuration (shared across all VLM backends)
    vlm=VLMConfig(
        backend="http-client",
        server_url="http://localhost:8000",
        max_concurrency=100,
        http_timeout=600
    )
)
```

## Preset Configurations

### Fast Processing

Optimized for speed - uses YOLO for layout, Fitz for text, skips tables:

```python
config = ParserConfig.fast()
```

### Accurate Processing

Uses VLM for all steps - best accuracy but requires VLM server:

```python
config = ParserConfig.accurate(vlm_server_url="http://localhost:8000")
```

### Scanned Documents

Optimized for scanned documents:

```python
config = ParserConfig.scanned(vlm_server_url="http://localhost:8000")
```

## Backend Options

### Layout Detection

| Backend | Description | Speed | Accuracy |
|---------|-------------|-------|----------|
| `none` | Skip layout detection | Fastest | N/A |
| `yolo` | YOLOv8 model | Fast | Good |
| `vlm` | Vision Language Model | Medium | Best |

### Table Parsing

| Backend | Description | Speed | Accuracy |
|---------|-------------|-------|----------|
| `none` | Skip table parsing | Fastest | N/A |
| `table_transformer` | Table Transformer | Medium | Good |
| `vlm` | Vision Language Model | Medium | Best |

### Text Extraction

| Backend | Description | Speed | Accuracy |
|---------|-------------|-------|----------|
| `none` | Skip text extraction | Fastest | N/A |
| `fitz` | Extract from PDF text layer | Fast | Best (digital PDFs) |
| `rapidocr` | RapidOCR engine | Medium | Good |
| `tesseract` | Tesseract OCR | Medium | Good |
| `vlm` | Vision Language Model | Medium | Best |
| `auto` | Auto-select (fitz for digital, OCR for scanned) | Varies | Best |

### Orientation Detection

| Backend | Description | Speed | Accuracy |
|---------|-------------|-------|----------|
| `none` | Skip orientation detection | Fastest | N/A |
| `tesseract` | Tesseract OSD | Fast | Good |
| `vlm` | VLM-based detection | Medium | Best |

### Skew Detection

| Backend | Description | Speed | Accuracy |
|---------|-------------|-------|----------|
| `none` | Skip skew detection | Fastest | N/A |
| `jdeskew` | Jdeskew library | Fast | Good |

## Usage Examples

### Example 1: Simple PDF Parsing

```python
from document_parser import ParserConfig, ConfigurablePDFProcessor

config = ParserConfig(device="cuda")
processor = ConfigurablePDFProcessor(config)
pages = processor.process("document.pdf")

# Export to markdown
from document_parser import ConfigurablePDFProcessor
ConfigurablePDFProcessor.convert_to_markdown(pages, "output.md")
```

### Example 2: VLM Pipeline

```python
from document_parser import ParserConfig, ConfigurablePDFProcessor

config = ParserConfig(
    device="cuda",
    layout=LayoutConfig(backend="vlm"),
    table=TableConfig(backend="vlm"),
    text=TextConfig(backend="vlm"),
    vlm=VLMConfig(server_url="http://localhost:8000")
)

processor = ConfigurablePDFProcessor(config)
pages = processor.process("document.pdf")
```

### Example 3: Mixed Backends

```python
from document_parser import ParserConfig, LayoutConfig, TableConfig, TextConfig

# YOLO for layout, VLM for tables, Auto for text
config = ParserConfig(
    device="cuda",
    layout=LayoutConfig(backend="yolo", confidence=0.6),
    table=TableConfig(backend="vlm"),
    text=TextConfig(backend="auto")
)

processor = ConfigurablePDFProcessor(config)
pages = processor.process("document.pdf")
```

### Example 4: Disable Specific Steps

```python
from document_parser import ParserConfig, TableConfig, OrientationConfig, SkewConfig

config = ParserConfig(
    table=TableConfig(enabled=False),
    orientation=OrientationConfig(enabled=False),
    skew=SkewConfig(enabled=False)
)

processor = ConfigurablePDFProcessor(config)
pages = processor.process("document.pdf")
```

## Architecture

```
document_parser/
├── __init__.py              # Package exports
├── config.py                # Configuration classes
├── interfaces.py            # Abstract backend interfaces
├── backend_factory.py       # Backend factory
├── processor.py             # ConfigurablePDFProcessor
└── backends/
    ├── __init__.py
    ├── layout/
    │   ├── yolo_backend.py
    │   ├── vlm_backend.py
    │   └── none_backend.py
    ├── table/
    │   ├── table_transformer_backend.py
    │   ├── vlm_backend.py
    │   └── none_backend.py
    ├── text/
    │   ├── fitz_backend.py
    │   ├── rapidocr_backend.py
    │   ├── tesseract_backend.py
    │   ├── vlm_backend.py
    │   ├── auto_backend.py
    │   └── none_backend.py
    ├── orientation/
    │   ├── tesseract_backend.py
    │   ├── vlm_backend.py
    │   └── none_backend.py
    └── skew/
        ├── jdeskew_backend.py
        └── none_backend.py
```

## Migration Guide

### From Legacy PDFProcessor

The new system is fully backward compatible:

```python
# Old code (still works)
from process import PDFProcessor
processor = PDFProcessor(device="cpu", zoom_factor=3)
pages = processor.process("document.pdf")

# New code (optional upgrade)
from document_parser import ParserConfig, ConfigurablePDFProcessor
config = ParserConfig(device="cpu", zoom_factor=3)
processor = ConfigurablePDFProcessor(config)
pages = processor.process("document.pdf")
```

## Requirements

- Python 3.8+
- PyMuPDF (fitz)
- Pillow
- ultralytics (for YOLO backend)
- rapidocr_onnxruntime (for RapidOCR backend)
- pytesseract (for Tesseract backend)
- jdeskew (for Jdeskew backend)
- mineru-vl-utils (for VLM backends)

## License

Part of the DocAI project.