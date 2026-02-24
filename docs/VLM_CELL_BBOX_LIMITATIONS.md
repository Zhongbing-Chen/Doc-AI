# Getting Cell-Level BBox from VLM

## Current Situation

**MinerU VLM (via mineru-vl-utils) provides:**
- ✅ Table-level bbox
- ✅ HTML table content
- ❌ **NO individual cell bbox**

## Why VLM Doesn't Provide Cell BBox

Most VLMs (including MinerU) are trained to:
1. **Detect** table regions (table bbox)
2. **Recognize** table structure and content
3. **Output** in structured format (HTML, Markdown, etc.)

They are **NOT** trained to output pixel-level cell coordinates because:
- Cell bbox is expensive to annotate in training data
- VLM focuses on content understanding, not geometric precision
- Different use cases need different bbox formats

---

## Solutions

### Solution 1: Use OCR to Get Cell BBox ⭐ **(Recommended)**

**Workflow:**
```
VLM → HTML Table Structure → OCR Text Boxes → Match & Refine → Cell BBox
```

**Implementation:**
```python
# 1. VLM gives HTML structure
html = "<table><tr><td>Cell 1</td>...</tr></table>"

# 2. Parse HTML to get grid structure
cells = parse_html_to_grid(html)

# 3. Run OCR to get text positions
ocr_boxes = run_ocr(image)  # [{'text': 'Cell 1', 'bbox': [x1,y1,x2,y2]}, ...]

# 4. Match OCR text to cell content
for cell in cells:
    matching_ocr = find_ocr_by_content(cell['content'], ocr_boxes)
    if matching_ocr:
        cell['bbox'] = merge_bboxes(matching_ocr)
    else:
        cell['bbox'] = estimate_from_grid(cell)
```

**Pros:**
- ✅ Uses actual text positions from OCR
- ✅ More accurate than grid estimation
- ✅ Works with any VLM that outputs HTML
- ✅ Can handle irregular cells

**Cons:**
- ⚠️ Requires OCR (adds processing time)
- ⚠️ Still needs matching logic

---

### Solution 2: Use Table Transformer Model (Most Accurate)

**Workflow:**
```
VLM → Detect Table Region → Table Transformer → Precise Cell BBox
```

**Implementation:**
```python
from table_transformer import TableTransformer

# 1. VLM detects table bbox
table_bbox = vlm.detect_table(image)

# 2. Crop table region
table_image = crop(image, table_bbox)

# 3. Table transformer detects cells
cells = table_transformer.detect_cells(table_image)
# Returns: [{'bbox': [x1,y1,x2,y2], 'row': 0, 'col': 0}, ...]
```

**Pros:**
- ✅ Most accurate (trained for cell detection)
- ✅ Handles complex layouts
- ✅ Provides precise boundaries

**Cons:**
- ❌ Requires additional model
- ❌ Slower (two-stage process)
- ❌ More complex pipeline

---

### Solution 3: Use Different VLM/Model

Some models provide cell-level output:

#### Option A: Table-Transformer (direct on image)
```python
from transformers import AutoModelForObjectDetection

model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
# Outputs cell bboxes directly
```

#### Option B: PaddleOCR + Table Structure
```python
from paddleocr import PPStructure

table_engine = PPStructure(show_log=True)
result = table_engine(image)
# Returns HTML + cell bbox
```

#### Option C: RapidOCR Table Recognition
```python
from rapidocr_onnxruntime import VisualModel

ocr = VisualModel()
result = ocr.detect(image)
# Can detect table structure with cell positions
```

---

## Recommendation

### For Production Use: Hybrid Approach

```python
def extract_table_with_cell_bbox(image):
    """Extract tables with cell bbox using VLM + OCR."""

    # Step 1: VLM detects tables and provides HTML structure
    vlm_blocks = vlm_client.detect(image)
    tables = [b for b in vlm_blocks if b.type == 'table']

    # Step 2: Run OCR to get all text positions
    ocr_boxes = ocr_engine.detect(image)

    # Step 3: For each table, refine cell bbox using OCR
    for table in tables:
        html = table.content
        table_bbox = table.bbox

        # Parse HTML to grid
        cells = parse_html_grid(html)

        # Refine with OCR
        for cell in cells:
            cell['bbox'] = refine_with_ocr(
                estimated_bbox=cell['grid_bbox'],
                content=cell['content'],
                ocr_boxes=ocr_boxes
            )

    return tables
```

---

## Comparison

| Method | Cell BBox Accuracy | Speed | Complexity |
|--------|-------------------|-------|-----------|
| **VLM Only (HTML)** | ❌ None | ⚡⚡⚡ | Low |
| **VLM + OCR** | ⭐⭐⭐⭐ | ⚡⚡ | Medium |
| **VLM + Table Transformer** | ⭐⭐⭐⭐⭐ | ⚡ | High |
| **Table Transformer Only** | ⭐⭐⭐⭐⭐ | ⚡ | Medium |
| **PaddleOCR** | ⭐⭐⭐⭐ | ⚡⚡ | Low |

---

## Conclusion

**Current MinerU VLM (via mineru-vl-utils) does NOT provide cell-level bbox.**

To get cell bbox, you must:
1. **Use OCR to refine positions** (recommended, what we implemented)
2. **Use Table Transformer** (most accurate)
3. **Switch to a different model** that outputs cell bbox directly

The VLM+OCR approach we implemented is the **best balance** of accuracy and speed!
