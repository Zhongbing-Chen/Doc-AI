# Process.py Usage Guide

## Quick Start

### Test with BOC PDF (Default)
```bash
# Uses scripts/boc.PDF as default test file
python process.py --use-mineru
```

### Process Your Own Files
```bash
# Process PDF with MinerU VLM
python process.py --use-mineru --file /path/to/document.pdf

# Process image with MinerU VLM
python process.py --use-mineru --image /path/to/image.png

# Use custom server URL
python process.py --use-mineru --server-url http://localhost:8000 --file doc.pdf

# Enable visualization (saves detection images)
python process.py --use-mineru --visualize --file doc.pdf
```

### Use Legacy Pipeline
```bash
# Use original modular pipeline (default)
python process.py --file document.pdf

# Specify device
python process.py --device cuda --file document.pdf
```

## Output

The script will generate:
- `list_output.md` - Markdown format content
- `json_output.json` - Structured JSON data
- `results/*/detail/*.png` - Visualization (if --visualize enabled)

## Example Output

```
======================================================================
Using MinerU VLM Parser
Server URL: http://localhost:8000
======================================================================

Input: /home/zhongbing/Projects/MLE/DocAI/scripts/boc.PDF
Type: PDF

  Processing 371 pages...
  Page 1/371... extracting... 6 blocks
  Page 2/371... extracting... 5 blocks
  Page 3/371... extracting... 3 blocks
  ...

======================================================================
✓ Processing Complete!
======================================================================
Time taken: 1234.56 seconds
Pages processed: 371
Total blocks: 1856

Block types:
  table: 45
  table_caption: 371
  text: 1440

✓ Markdown saved to: list_output.md
✓ JSON saved to: json_output.json
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--use-mineru` | Enable MinerU VLM parser | False (uses legacy) |
| `--file` | Path to PDF file | scripts/boc.PDF |
| `--image` | Path to image file | img.png |
| `--server-url` | MinerU VLM server URL | http://localhost:8000 |
| `--visualize` | Enable visualization | False |
| `--device` | Device for legacy pipeline | cpu |

## Tips

1. **Start vLLM server first:**
   ```bash
   vllm serve /path/to/MinerU2.5-2509-1.2B --port 8000
   ```

2. **Disable proxy for localhost:**
   ```bash
   export NO_PROXY=localhost
   ```

3. **For large PDFs:** Processing time scales with page count (approximately 3-10 seconds per page)

4. **Monitor progress:** The script shows page-by-page progress with block counts

5. **Test first:** Use `--image` with a single page image to test before processing large PDFs
