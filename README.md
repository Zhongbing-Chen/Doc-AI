# Document Parser

A unified document parsing system that converts various document formats into a consistent JSON structure. Supports PDF, Word, Excel, PowerPoint, Markdown, and Images with optional VLM (Vision Language Model) enhancement.

## Features

- **Uniform JSON Output**: All document types are converted to the same JSON schema
- **Multiple Format Support**: PDF, Word (.doc/.docx), Excel (.xls/.xlsx), PowerPoint (.ppt/.pptx), Markdown, Images
- **VLM Integration**: Optional multimodal VLM parsing via vLLM for enhanced document understanding
- **YOLO Layout Detection**: Fast layout detection with YOLOv8 (uses local model file)
- **Block-level Structure**: Extract text, tables, figures with bounding boxes
- **Markdown Export**: Automatic conversion to markdown format

## Installation

```bash
# Install core dependencies
pip install fitz pillow markitdown

# Install with Office document support
pip install 'markitdown[docx,pdf,pptx]'

# Install with YOLO layout detection
pip install ultralytics

# Install with VLM support
pip install httpx
```

## Quick Start

### Parse Any Document

```python
from document_parser import DocumentParser

# Create parser
parser = DocumentParser()

# Parse any document
doc = parser.parse("document.pdf")  # or .docx, .xlsx, .pptx, .md, .png

# Get JSON output
json_str = doc.to_json(indent=2)

# Save to file
doc.save("output.json")
```

### Parse with VLM Enhancement

```python
# Start vLLM server first
# vllm serve llava-hf/llava-v1.6-mistral-7b-hf --port 8000

# Create parser with VLM
parser = DocumentParser(vlm_url="http://localhost:8000")

# Parse with VLM enhancement
doc = parser.parse("scanned_document.pdf")
print(doc.to_json())
```

## Uniform JSON Format

All documents are converted to this consistent schema:

```json
{
  "version": "1.0",
  "metadata": {
    "file_name": "document.pdf",
    "file_path": "/path/to/document.pdf",
    "file_size": 102400,
    "file_type": "pdf",
    "file_extension": ".pdf",
    "mime_type": "application/pdf",
    "processed_at": "2024-01-01T00:00:00Z"
  },
  "document_info": {
    "total_pages": 10,
    "title": "Document Title",
    "author": "Author Name"
  },
  "pages": [
    {
      "page_num": 1,
      "width": 612,
      "height": 792,
      "rotation": 0,
      "blocks": [
        {
          "block_id": "p1_b001",
          "type": "paragraph",
          "bbox": {
            "x_1": 50,
            "y_1": 100,
            "x_2": 550,
            "y_2": 150,
            "coordinate_system": "pixels",
            "origin": "top_left"
          },
          "content": {
            "text": "Text content",
            "markdown": "**Markdown** content"
          },
          "confidence": 0.95,
          "properties": {
            "font_size": 12
          }
        }
      ],
      "page_content": "Full page text"
    }
  ],
  "full_text": "Complete document text",
  "full_markdown": "# Complete document markdown"
}
```

## Block Types

| Type | Description |
|------|-------------|
| `paragraph` | Regular text content |
| `title` | Document title |
| `heading_1`, `heading_2`, `heading_3` | Section headings |
| `list`, `list_item` | Lists |
| `table` | Table with structured data |
| `figure` | Image or figure |
| `equation` | Mathematical equation |
| `code` | Code block |
| `quote` | Quoted text |

## Table Data Structure

Tables include structured cell information:

```json
{
  "type": "table",
  "table_data": {
    "rows": 3,
    "cols": 3,
    "cells": [
      {
        "row": 0,
        "col": 0,
        "row_span": 1,
        "col_span": 1,
        "content": "Header",
        "is_header": true
      }
    ]
  }
}
```

## API Reference

### DocumentParser

```python
class DocumentParser:
    def __init__(self, vlm_url: Optional[str] = None):
        """
        Args:
            vlm_url: vLLM server URL (default: None)
                     Example: "http://localhost:8000"
        """

    def parse(self, file_path: str) -> Document:
        """Parse document to uniform JSON format"""

    def parse_to_json(self, file_path: str, output_path: str = None) -> str:
        """Parse document and return JSON string"""
```

### Document

```python
class Document:
    version: str                    # Schema version
    metadata: DocumentMetadata      # File metadata
    document_info: DocumentInfo     # Document info
    pages: List[Page]              # Pages with blocks
    full_text: str                 # Complete text
    full_markdown: str             # Complete markdown

    def to_json(self, indent: int = 2) -> str
    def save(self, file_path: str) -> None

    @classmethod
    def from_json(cls, json_str: str) -> 'Document'
    @classmethod
    def from_file(cls, file_path: str) -> 'Document'
```

## vLLM Integration

### Starting vLLM Server

```bash
# Start vLLM with LLaVA model
vllm serve llava-hf/llava-v1.6-mistral-7b-hf \
    --port 8000 \
    --trust-remote-code

# Or with Qwen-VL
vllm serve Qwen/Qwen2-VL-7B-Instruct \
    --port 8000 \
    --trust-remote-code
```

### Using VLM Parser

```python
from document_parser import DocumentParser

# Connect to vLLM server
parser = DocumentParser(vlm_url="http://localhost:8000")

# Parse with VLM
doc = parser.parse("complex_document.pdf")

# VLM provides:
# - Better layout detection
# - Accurate table parsing
# - Figure descriptions
# - Equation recognition
```

## YOLO Layout Detection

The YOLO backend uses a local model file by default:

```python
from document_parser import DocumentParser
from document_parser.config import ParserConfig, LayoutConfig

# Configure with local YOLO model
config = ParserConfig(
    layout=LayoutConfig(
        backend="yolo",
        model_path="/path/to/best.pt",  # Local model file
        confidence=0.5
    )
)
```

## Examples

### Parse Word Document

```python
from document_parser import DocumentParser

parser = DocumentParser()
doc = parser.parse("report.docx")

print(f"Title: {doc.document_info.title}")
print(f"Pages: {doc.document_info.total_pages}")
print(f"Blocks: {sum(len(p.blocks) for p in doc.pages)}")

# Save JSON
doc.save("report.json")
```

### Parse Excel Spreadsheet

```python
parser = DocumentParser()
doc = parser.parse("data.xlsx")

for page in doc.pages:
    print(f"Sheet: {page.page_num}")
    for block in page.blocks:
        if block.type == "table":
            print(f"  Table: {block.table_data['rows']}x{block.table_data['cols']}")
```

### Parse PowerPoint

```python
parser = DocumentParser()
doc = parser.parse("presentation.pptx")

print(f"Slides: {doc.document_info.total_pages}")
for page in doc.pages:
    print(f"Slide {page.page_num}: {len(page.blocks)} blocks")
```

### Parse PDF with VLM

```python
parser = DocumentParser(vlm_url="http://localhost:8000")
doc = parser.parse("scanned_research_paper.pdf")

# VLM extracts:
# - Text with proper formatting
# - Table structures
# - Figure captions
# - Equations

doc.save("paper.json")
```

### Batch Processing

```python
from pathlib import Path
from document_parser import DocumentParser

parser = DocumentParser()

for file_path in Path("documents").glob("*.pdf"):
    doc = parser.parse(str(file_path))
    doc.save(f"output/{file_path.stem}.json")
```

## Project Structure

```
document_parser/
├── __init__.py           # Package exports
├── format.py             # JSON format classes (Document, Page, Block)
├── parser.py             # Main DocumentParser
├── universal_parser.py   # MarkItDown-based parser
├── config.py             # Configuration classes
├── processor.py          # PDF processor (legacy)
├── backends/             # Backend implementations
│   └── layout/
│       └── yolo_backend.py
└── util/                 # Utilities
```

## Requirements

- Python 3.8+
- fitz (PyMuPDF)
- Pillow
- markitdown (for Office documents)
- ultralytics (optional, for YOLO)
- httpx (optional, for VLM)
- vllm (for VLM serving)

## License

Part of the DocAI project.
