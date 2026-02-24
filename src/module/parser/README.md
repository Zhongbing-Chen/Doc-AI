# Parser Module

Unified facade for document parsing with support for multiple strategies.

## Overview

This module provides a clean, unified interface for parsing PDF and image documents using two different strategies:

1. **MinerU VLM** (default): Vision-Language Model based parsing via HTTP client
2. **Legacy**: Modular pipeline with layout detection, table parsing, and text extraction

## Quick Start

### Simple Usage (Recommended)

```python
from module.parser import parse_pdf

# Parse PDF with default settings
pages = parse_pdf("document.pdf")

# Parse with custom configuration
pages = parse_pdf(
    "document.pdf",
    server_url="http://localhost:8000",
    max_workers=8
)
```

### Factory Pattern (Reusable Parser)

```python
from module.parser import ParserFactory

# Create a parser instance
parser = ParserFactory.create_parser(
    strategy="mineru_vlm",
    server_url="http://localhost:8000",
    zoom_factor=3.0,
    max_workers=4
)

# Use for multiple files
pdf_pages = parser.process_pdf("doc1.pdf")
img_pages = parser.process_image("scan1.png")
```

### Direct Import (Advanced)

```python
from module.parser import MinerUVLMParser

parser = MinerUVLMParser(server_url="http://localhost:8000")
pages = parser.process_pdf("document.pdf")
```

## API Reference

### Convenience Functions

#### `parse_pdf(file_path, strategy="mineru_vlm", **kwargs)`

Parse a PDF file using the specified strategy.

**Parameters:**
- `file_path` (str): Path to PDF file
- `strategy` (str): "mineru_vlm" (default) or "legacy"
- `**kwargs`: Strategy-specific configuration

**Returns:** `List[Page]` - List of Page objects

**Example:**
```python
pages = parse_pdf("document.pdf", strategy="mineru_vlm")
```

#### `parse_image(image_path, strategy="mineru_vlm", **kwargs)`

Parse an image file using the specified strategy.

**Parameters:**
- `image_path` (str): Path to image file
- `strategy` (str): "mineru_vlm" (default) or "legacy"
- `**kwargs`: Strategy-specific configuration

**Returns:** `List[Page]` - List of Page objects (typically 1)

**Example:**
```python
pages = parse_image("scan.png", strategy="mineru_vlm")
```

### ParserFactory

#### `ParserFactory.create_parser(strategy="mineru_vlm", **kwargs)`

Create a parser instance based on the specified strategy.

**Parameters:**
- `strategy` (str): "mineru_vlm" (default) or "legacy"
- `**kwargs`: Strategy-specific configuration

**Returns:** Parser instance with `process_pdf()` and `process_image()` methods

**Raises:** `ValueError` if strategy is not supported

### Configuration Options

**MinerU VLM parameters:**
- `server_url` (str): URL of vLLM server (default: "http://localhost:8000")
- `zoom_factor` (float): Zoom factor for PDF conversion (default: 3.0)
- `max_workers` (int): Concurrent workers (default: 4)
- `output_format` (str): "content_blocks", "html", or "both"
- `extract_cell_bbox` (bool): Extract cell-level bbox for tables (default: False)

**Legacy parameters:**
- `device` (str): "cpu" or "cuda"
- `zoom_factor` (int): Zoom factor for PDF conversion (default: 3)
- `model_source` (str): Model source (default: "huggingface")
- `visualize` (bool): Enable visualization (default: False)

## Usage Examples

### Example 1: Simple PDF Parsing

```python
from module.parser import parse_pdf

# Parse with defaults
pages = parse_pdf("document.pdf")

# Access results
for page in pages:
    print(f"Page {page.page_num}: {len(page.blocks)} blocks")
    for block in page.blocks:
        print(f"  [{block.label}] {block.content[:50]}...")
```

### Example 2: Batch Processing

```python
from module.parser import ParserFactory

# Create parser once
parser = ParserFactory.create_parser(
    strategy="mineru_vlm",
    server_url="http://localhost:8000",
    max_workers=8
)

# Process multiple files
pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]

for pdf_file in pdf_files:
    print(f"Processing {pdf_file}...")
    try:
        pages = parser.process_pdf(pdf_file)
        print(f"  ✓ {len(pages)} pages")
    except Exception as e:
        print(f"  ✗ Error: {e}")
```

### Example 3: Table Cell Extraction

```python
from module.parser import parse_pdf

# Parse with cell bbox extraction enabled
parser = ParserFactory.create_parser(
    strategy="mineru_vlm",
    extract_cell_bbox=True
)

pages = parser.process_pdf("document.pdf")

# Access table cell information
for page in pages:
    for block in page.blocks:
        if block.label == "table" and block.table_cells:
            print(f"Table with {len(block.table_cells)} cells")
            for cell in block.table_cells:
                print(f"  Cell[{cell['row']},{cell['col_start']}]: {cell['content']}")
```

### Example 4: Export to Different Formats

```python
from module.parser import parse_pdf
from process import PDFProcessor

# Parse document
pages = parse_pdf("document.pdf")

# Export to Markdown
markdown_content = PDFProcessor.convert_to_markdown(pages)
# Output saved to: list_output.md

# Export to JSON
json_content = PDFProcessor.merge(pages)
# Output saved to: json_output.json
```

### Example 5: Strategy Comparison

```python
from module.parser import ParserFactory, ParserStrategy

file_path = "document.pdf"

# Parse with MinerU VLM (recommended)
vlm_pages = ParserFactory.create_parser(
    strategy=ParserStrategy.MINERU_VLM.value
).process_pdf(file_path)

print(f"MinerU VLM: {len(vlm_pages)} pages")

# Parse with Legacy pipeline
legacy_pages = ParserFactory.create_parser(
    strategy=ParserStrategy.LEGACY.value,
    device="cpu"
).process_pdf(file_path)

print(f"Legacy: {len(legacy_pages)} pages")
```

## Page Object Structure

Each `Page` object contains:

- `page_num` (int): Page number (0-indexed)
- `blocks` (List[Block]): List of detected blocks
- `image` (PIL.Image): Page image
- `pdf_page` (fitz.Page): Original PDF page object
- `rotated_angle` (float): Rotation angle applied
- `skewed_angle` (float): Deskew angle applied
- `is_scanned` (bool): Whether the page is scanned

Each `Block` object contains:

- `block_id` (str): Unique identifier
- `x_1, y_1, x_2, y_2` (float): Bounding box coordinates
- `label` (str): Block type (text, title, table, etc.)
- `label_id` (int): Numeric label ID
- `content` (str): Extracted content (HTML tables auto-converted to Markdown)
- `table_cells` (List[dict], optional): Cell-level information for tables
- `table_structure` (List[TableStructure], optional): Detailed table structure

## Strategy Comparison

| Feature | MinerU VLM | Legacy |
|---------|-----------|--------|
| **Speed** | Faster (concurrent) | Slower (sequential) |
| **Accuracy** | Higher (VLM) | Good (rule-based) |
| **Tables** | Excellent (merged cells) | Good (structure detection) |
| **Resources** | External server required | Local CPU/GPU |
| **Configuration** | Simple | More options |
| **Use Case** | Production, complex docs | Debugging, custom needs |

## Architecture

```
module/parser/
├── __init__.py              # Facade - unified interface
├── mineru_vlm_parser.py     # MinerU VLM implementation
└── README.md                # This file
```

The facade pattern provides:
- **Single entry point**: `from module.parser import parse_pdf`
- **Strategy abstraction**: Switch between parsers via `strategy` parameter
- **Clean API**: Consistent interface regardless of underlying implementation
- **Extensibility**: Easy to add new parsing strategies

## Error Handling

```python
from module.parser import parse_pdf, ParserStrategy

try:
    pages = parse_pdf("document.pdf", strategy="mineru_vlm")
except ValueError as e:
    print(f"Configuration error: {e}")
except ConnectionError:
    print("Cannot connect to MinerU VLM server")
except Exception as e:
    print(f"Parsing error: {e}")
```

## Installation

The module is part of the DocAI project. Ensure dependencies are installed:

```bash
pip install -r requirements.txt
```

For MinerU VLM, ensure the vLLM server is running:

```bash
# Start MinerU VLM server
python -m mineru_vl_utils.server --port 8000
```

## Testing

Run a simple test:

```python
from module.parser import parse_pdf

pages = parse_pdf("test.pdf")
print(f"Parsed {len(pages)} pages")
```

## Common Patterns

### Simple One-Off Parsing

```python
from module.parser import parse_pdf

# Parse a single file
pages = parse_pdf("document.pdf")
print(f"Parsed {len(pages)} pages")
```

### Batch Processing Multiple Files

```python
from module.parser import ParserFactory
from pathlib import Path

# Create parser once for reuse
parser = ParserFactory.create_parser(
    strategy="mineru_vlm",
    max_workers=8
)

# Process multiple files
pdf_dir = Path("/path/to/pdfs")
for pdf_file in pdf_dir.glob("*.pdf"):
    print(f"Processing {pdf_file.name}...")
    try:
        pages = parser.process_pdf(str(pdf_file))
        print(f"  ✓ {len(pages)} pages")
    except Exception as e:
        print(f"  ✗ Error: {e}")
```

### Processing Images

```python
from module.parser import parse_image

# Parse single image
pages = parse_image("scan.png")
print(f"Extracted {len(pages[0].blocks)} blocks")

# Parse multiple images
for img_file in Path("scans/").glob("*.png"):
    pages = parse_image(str(img_file))
    print(f"{img_file.name}: {len(pages[0].blocks)} blocks")
```

### Exporting Results

```python
from module.parser import parse_pdf
from process import PDFProcessor

# Parse document
pages = parse_pdf("document.pdf")

# Export to Markdown
markdown_content = PDFProcessor.convert_to_markdown(pages)
# Saves to: list_output.md

# Export to JSON
json_content = PDFProcessor.merge(pages)
# Saves to: json_output.json

# Access data programmatically
for page in pages:
    for block in page.blocks:
        if block.label == "table":
            print(f"Found table with content:")
            print(block.content)
```

### Working with Tables

```python
from module.parser import ParserFactory

# Enable cell bbox extraction for detailed table info
parser = ParserFactory.create_parser(
    strategy="mineru_vlm",
    extract_cell_bbox=True
)

pages = parser.process_pdf("financial_report.pdf")

# Access table cells
for page in pages:
    for block in page.blocks:
        if block.label == "table" and block.table_cells:
            print(f"Table with {len(block.table_cells)} cells:")
            for cell in block.table_cells:
                row, col = cell['row'], cell['col_start']
                content = cell['content']
                print(f"  [{row},{col}]: {content}")
```

### Error Handling

```python
from module.parser import parse_pdf
import sys

files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]

for file_path in files:
    try:
        pages = parse_pdf(file_path)
        print(f"✓ {file_path}: {len(pages)} pages")
    except ValueError as e:
        print(f"✗ {file_path}: Invalid configuration - {e}")
    except ConnectionError:
        print(f"✗ {file_path}: Cannot connect to VLM server")
        sys.exit(1)
    except Exception as e:
        print(f"✗ {file_path}: {e}")
```

### Comparing Strategies

```python
from module.parser import ParserFactory, ParserStrategy

file_path = "document.pdf"

# MinerU VLM (recommended for production)
vlm_parser = ParserFactory.create_parser(
    strategy=ParserStrategy.MINERU_VLM.value
)
vlm_pages = vlm_parser.process_pdf(file_path)
print(f"VLM: {len(vlm_pages)} pages, {sum(len(p.blocks) for p in vlm_pages)} blocks")

# Legacy (for debugging/custom needs)
legacy_parser = ParserFactory.create_parser(
    strategy=ParserStrategy.LEGACY.value,
    device="cpu"
)
legacy_pages = legacy_parser.process_pdf(file_path)
print(f"Legacy: {len(legacy_pages)} pages, {sum(len(p.blocks) for p in legacy_pages)} blocks")
```

## Contributing

When adding a new parsing strategy:

1. Create a new parser class (e.g., `NewParser.py`)
2. Add strategy to `ParserStrategy` enum in `__init__.py`
3. Add factory creation logic in `ParserFactory.create_parser()`
4. Update documentation

## License

Part of the DocAI project.
