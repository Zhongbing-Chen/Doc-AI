# VLM HTML Table Cell BBox Validation Script

## Overview

Validates that MinerU VLM can:
1. ✅ Extract tables in HTML format
2. ✅ Parse HTML to get cell-level structure
3. ✅ Calculate cell bbox coordinates
4. ✅ Convert HTML to Markdown
5. ✅ Visualize cell boundaries

---

## Usage

### Basic Usage

```bash
# Validate with BOC PDF (page 3 has tables)
python scripts/validate_vlm_cell_bbox.py /path/to/document.pdf

# Validate with image
python scripts/validate_vlm_cell_bbox.py /path/to/image.png
```

### With Custom Page Number

```python
# Edit script to change page_num, or:
python -c "
from scripts.validate_vlm_cell_bbox import visualize_cells_with_bbox
visualize_cells_with_bbox('/path/to/file.pdf', page_num=5)
"
```

---

## What It Does

### 1. Extracts Tables from VLM
```
VLM Server → HTML Table String
```

### 2. Parses HTML Structure
```python
html = "<table><tr><td>Cell 1</td><td>Cell 2</td></tr>...</table>"

# Parse to get:
cells = [
    {
        'content': 'Cell 1',
        'row': 0,
        'col_start': 0,
        'col_end': 1,
        'bbox': [134.0, 280.4, 893.0, 353.5],
        'bbox_source': 'estimated'
    },
    ...
]
```

### 3. Converts to Markdown
```markdown
| Cell 1 | Cell 2 |
|---|---|
```

### 4. Visualizes Results
- Draws table boundary (red)
- Draws cell boundaries (blue/green/cyan/orange)
- Numbers each cell
- Saves as PNG image

---

## Output Files

### 1. Visualization Image: `table_1_cells_visualization.png`
Shows:
- Table bounding box (red)
- Cell boundaries (colored by position)
- Cell numbers

### 2. Cell Data: `table_1_cells.json`
Contains:
```json
{
  "table_bbox": [x1, y1, x2, y2],
  "html_content": "<table>...</table>",
  "markdown_content": "| H1 | H2 |...",
  "cells": [
    {
      "content": "Cell text",
      "row": 0,
      "col_start": 0,
      "col_end": 1,
      "colspan": 1,
      "rowspan": 1,
      "bbox": [x1, y1, x2, y2],
      "bbox_source": "estimated"
    },
    ...
  ]
}
```

---

## Example Output

```
================================================================================
VLM HTML Table Extraction with Cell BBox Validation
================================================================================

1. Loading image: /path/to/boc_first_10_pages.pdf
   ✓ Extracted page 3 from PDF
   Image size: 1786 x 2526

2. Connecting to VLM server: http://localhost:8000
   ✓ Connected to VLM server

3. Extracting tables using VLM...
   ✓ Found 3 blocks
   ✓ Found 1 tables

================================================================================
Table 1
================================================================================
Table bbox: [133.9, 280.4, 1652.1, 2109.2]
HTML length: 1314 characters

4. Parsing HTML table structure (grid estimation)...
   ✓ Extracted 45 cells

   First 5 cells:
     Cell 1: Row 0, Col 0-1
       Content: The Banker(《银行家》)...
       BBox: [133.9, 280.4, 893.0, 353.5]
       Span: colspan=1, rowspan=1

5. Converting HTML to Markdown...
   ✓ Markdown table generated

6. Visualizing table with cell boundaries...
   ✓ Saved visualization to: table_1_cells_visualization.png
   ✓ Saved cell data to: table_1_cells.json

================================================================================
✓ Validation complete!
================================================================================
```

---

## Validation Checklist

Run this script to verify:

- ✅ VLM server is running and accessible
- ✅ VLM can detect tables in images/PDFs
- ✅ VLM outputs tables in HTML format
- ✅ HTML can be parsed to get cell structure
- ✅ Cell bbox can be calculated from table bbox
- ✅ HTML tables can be converted to Markdown
- ✅ Cell boundaries can be visualized

---

## Prerequisites

### VLM Server
```bash
# Start vLLM server
vllm serve /path/to/MinerU-model \
  --gpu-memory-utilization 0.3 \
  --port 8000
```

### Python Dependencies
```bash
# Install required packages
pip install mineru-vl-utils
pip install pillow
pip install pymupdf
```

---

## Troubleshooting

### No tables found
```
   ✗ No tables found in image!
```
**Solution:** Try a different page or image with visible tables

### Connection failed
```
   ✗ Failed to connect: ...
```
**Solution:** Start vLLM server first
```bash
vllm serve /path/to/model --gpu-memory-utilization 0.3
```

### Cell bbox looks wrong
**Note:** This script uses grid estimation (not OCR refinement).
For more accurate bbox, use `extract_cell_bbox=True` in main pipeline.

---

## Next Steps

### For Grid Estimation (Current Script)
```bash
python scripts/validate_vlm_cell_bbox.py document.pdf
```

### For OCR-Refined Cell BBox (More Accurate)
```python
from process import PDFProcessor

processor = PDFProcessor(use_mineru_vlm=True, ...)
processor.mineru_parser.extract_cell_bbox = True
pages = processor.process("document.pdf")

# Access refined cell bbox
for cell in block.table_cells:
    print(f"BBox: {cell['bbox']}")
    print(f"Source: {cell['bbox_source']}")  # 'ocr_refined'
```

---

## Summary

**This validation script proves:**
1. ✅ VLM successfully extracts tables in HTML format
2. ✅ HTML contains table structure (rows, cols, spans)
3. ✅ Cell bbox can be calculated from table bbox
4. ✅ HTML → Markdown conversion works correctly
5. ✅ Cell boundaries can be visualized

**The VLM → HTML → Cell BBox pipeline is fully functional!** 🎉
