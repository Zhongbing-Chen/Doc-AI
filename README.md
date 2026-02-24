# DocAI - Document AI Parser

A document parsing and analysis system powered by VLM (Vision Language Models) and traditional OCR/table recognition techniques.

## Project Structure

```
DocAI/
├── src/                    # Source code (all you need for development)
│   ├── entity/            # Data entities (block, box, page)
│   ├── module/            # Processing modules
│   │   ├── layout/       # Layout detection
│   │   ├── text/         # Text recognition (OCR)
│   │   ├── table/        # Table parsing
│   │   └── rotation/     # Orientation correction
│   ├── model/             # Model definitions
│   │   └── table_transformer/  # Table Transformer model
│   ├── mineru_client/     # MinerU API client
│   ├── mineru_server/     # MinerU server implementation
│   ├── frontend/          # Web frontend (Flask + vanilla JS)
│   └── util/              # Utility functions
├── assets/                # Large files (models, PDFs, images - not tracked)
├── docs/                  # Documentation
└── .gitignore            # Git ignore rules
```

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js (optional, for some frontend features)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Zhongbing-Chen/DocAI-UI.git
cd DocAI-UI
```

2. Download model weights (if needed):
- Table Transformer models
- OCR models (PaddleOCR)

3. Install dependencies:
```bash
pip install -r src/requirements.txt
```

### Running the Frontend

```bash
cd src/frontend
python server.py
```

Then open http://localhost:5002 in your browser.

## Features

- Document layout analysis
- OCR text extraction
- Table recognition and parsing
- Coordinate mapping between PDF and markdown
- Interactive document viewer
- MinerU integration

## Documentation

See the [docs/](docs/) directory for detailed documentation:

- [MinerU Quick Start](docs/MINERU_QUICK_START.md)
- [VLM Integration](docs/mineru_vlm_integration_architecture.md)
- [OCR Cell BBox](docs/OCR_CELL_BBOX.md)
- [Table Recognition](docs/VLM_TABLE_RECOGNITION_TEST.md)

## Note on Large Files

This repository excludes large model files (>50MB) to comply with GitHub's file size limits. Model files should be downloaded separately from their official sources:

- Table Transformer: HuggingFace Microsoft/table-transformer
- OCR Models: PaddleOCR official release

## License

[Add your license here]
