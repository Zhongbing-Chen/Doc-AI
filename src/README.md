# Source directory for DocAI

This directory contains all the source code for the DocAI project.

## Structure

- `entity/` - Data entities (block, box, page)
- `module/` - Processing modules
  - `layout/` - Layout detection
  - `text/` - Text recognition
  - `table/` - Table parsing
  - `rotation/` - Orientation correction
- `model/` - Model definitions
  - `table_transformer/` - Table Transformer model
  - `ocr/` - OCR model files
- `mineru_client/` - MinerU API client
- `mineru_server/` - MinerU server implementation
- `frontend/` - Web frontend
- `util/` - Utility functions
- `process.py` - Main processing script
- `requirements.txt` - Python dependencies
