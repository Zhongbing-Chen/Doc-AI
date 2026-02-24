# MinerU VLM Integration - Quick Reference

## Quick Start

### 1. Start vLLM Server
```bash
vllm serve /path/to/MinerU2.5-2509-1.2B \
  --host 127.0.0.1 \
  --port 8000 \
  --logits-processors mineru_vl_utils:MinerULogitsProcessor
```

### 2. Use in Python
```python
from process import PDFProcessor

# Enable MinerU VLM
processor = PDFProcessor(
    use_mineru_vlm=True,
    mineru_server_url="http://localhost:8000"
)

# Process
pages = processor.process_img("img.png")
# or
pages = processor.process("document.pdf")

# Output
PDFProcessor.convert_to_markdown(pages)  # → list_output.md
PDFProcessor.merge(pages)                 # → json_output.json
```

### 3. Test
```bash
python test_mineru_integration.py --image img.png
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_mineru_vlm` | bool | False | Enable MinerU VLM parser |
| `mineru_server_url` | str | "http://localhost:8000" | vLLM server URL |
| `zoom_factor` | float | 3.0 | PDF page zoom |
| `visualize` | bool | False | Save visualization |

## Examples

### Process Image
```python
processor = PDFProcessor(use_mineru_vlm=True)
pages = processor.process_img("document.png")
```

### Process PDF with Visualization
```python
processor = PDFProcessor(
    use_mineru_vlm=True,
    visualize=True
)
pages = processor.process("document.pdf")
```

### Custom Server URL
```python
processor = PDFProcessor(
    use_mineru_vlm=True,
    mineru_server_url="http://192.168.1.100:8000"
)
```

### Legacy Pipeline (Default)
```python
# No changes needed - this uses original pipeline
processor = PDFProcessor()
pages = processor.process("document.pdf")
```

## Output

Each page contains `blocks` with:
- `label` - Type (text, table, equation, etc.)
- `x_1, y_1, x_2, y_2` - Bounding box coordinates
- `content` - Extracted text/HTML/LaTeX

## Files

- `process.py` - Updated with MinerU support
- `module/parser/mineru_vlm_parser.py` - MinerU parser implementation
- `test_mineru_integration.py` - Test script
- `scripts/validate_mineru_client.py` - Validation script

## Troubleshooting

**Connection Error**:
```bash
# Check server is running
curl http://localhost:8000/health

# Disable proxy
export NO_PROXY=localhost
```

**Import Error**:
```bash
pip install mineru-vl-utils
```

## That's It!

Just set `use_mineru_vlm=True` and you're ready to go! 🚀
