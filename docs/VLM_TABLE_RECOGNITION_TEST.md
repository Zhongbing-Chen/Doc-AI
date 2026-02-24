# VLM Table Recognition Test Results

## Test Summary

Tested whether MinerU VLM's "Table Recognition" mode provides cell-level bbox coordinates.

## Result: ❌ NO Cell-Level BBox in Table Recognition

### What VLM Table Recognition Provides

When using the "Table Recognition" prompt on a cropped table image, VLM returns output in OTSL (Optical Table Structure Language) format:

```
<fcel>The Banker (《银行家》)<fcel>全球银行 1000 强第 4 位<nl>
<fcel>The Banker (《银行家》)、
Brand Finance (《品牌金融》)<fcel>全球银行品牌价值 500 强第 4 位<nl>
...
```

**Tags:**
- `<fcel>` = First cell in a row
- `<ucel>` = Cell that continues from above row (rowspan > 1)
- `<nl>` = New line (new row)
- Content text

**What it DOES provide:**
- ✅ Cell structure (which cells exist)
- ✅ Merged cell information (via `<ucel>` for rowspan)
- ✅ Cell content (text)
- ✅ Row/column organization

**What it DOES NOT provide:**
- ❌ NO bbox coordinates for cells
- ❌ NO `<|box_start|>` tokens like in Layout Detection
- ❌ NO spatial position of cells within the table

## Test Details

### Input
- Image: Page 3 from boc_first_10_pages.pdf
- Table size: 1518 x 1829 pixels (cropped)
- Table bbox: [133.9, 280.4, 1652.1, 2109.2] (original image coordinates)

### Process
1. **Layout Detection**: Found 1 table at bbox [0.075, 0.111, 0.925, 0.835] (normalized)
2. **Crop Table**: Cropped table region from original image
3. **Table Recognition**: Called VLM with "\nTable Recognition:" prompt
4. **Parse Output**: Searched for `<|box_start|>` tokens

### Output
- Total lines: 28
- Found 0 elements with bbox
- Output format: OTSL with `<fcel>`, `<ucel>`, `<nl>` tags
- NO bbox coordinates found

## Comparison

| Mode | Provides BBox | Format | What You Get |
|------|---------------|--------|--------------|
| **Layout Detection** | ✅ Yes | `<|box_start|>x1 y1 x2 y2<|box_end|><|ref_start|>type<|ref_end|>content` | Block-level bbox (table, text, title) |
| **Table Recognition** | ❌ NO | `<fcel>content<fcel>content<nl>` | Cell structure and content only |

## Conclusion

**MinerU VLM does NOT provide cell-level bbox coordinates in either mode:**
- Layout Detection: Only gives table-level bbox (not cells)
- Table Recognition: Only gives cell structure and content (no bbox)

## To Get Cell BBox

You still need to use one of these approaches:

1. **Grid Estimation** (fastest, least accurate)
   - Calculate cell bbox from table bbox and HTML grid structure
   - Use `parse_html_table_structure()` with `ocr_boxes=None`

2. **OCR Refinement** (recommended, good balance)
   - Run OCR to get text positions
   - Match OCR text to cell content
   - Use `parse_html_table_structure()` with `ocr_boxes=ocr_result`

3. **Table Transformer** (most accurate, slowest)
   - Use dedicated table structure model
   - Provides precise cell bbox

## Test Script

Location: `util/test_vlm_table_recognition.py`

Run with:
```bash
NO_PROXY=localhost python util/test_vlm_table_recognition.py
```

## Technical Details

### VLM Output Format for Tables

**Layout Detection Mode:**
```
<|box_start|>092 071 283 100<|box_end|><|ref_start|>table<|ref_end|>
```
- Provides bbox for entire table
- Format: 0-1000 range coordinates

**Table Recognition Mode:**
```
<fcel>Cell 1<fcel>Cell 2<nl>
<fcel>Cell 3<ucel><nl>
```
- NO bbox coordinates
- Only cell markers and content
- OTSL format (converted to HTML by mineru-vl-utils)

### Code Reference

The OTSL parser is in:
`/home/zhongbing/.local/lib/python3.10/site-packages/mineru_vl_utils/post_process/otsl2html.py`

It converts OTSL format to HTML, but does NOT add bbox information.

---

**Date**: 2024-12-27
**Tested By**: Claude Code
**VLM Version**: MinerU (via mineru-vl-utils)
