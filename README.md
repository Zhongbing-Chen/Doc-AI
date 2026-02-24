# DocAI - Document AI Parser

A document parsing and analysis system powered by VLM (Vision Language Models) and traditional OCR/table recognition techniques.

## Project Structure

```
DocAI/
├── src/
│   ├── mineru_client/     # MinerU API client
│   ├── mineru_server/     # MinerU server implementation
│   └── frontend/          # Web frontend (Flask + vanilla JS)
├── model/                 # Model definitions (table transformer, etc.)
├── module/                # Processing modules (layout, text, table, rotation)
├── docs/                  # Documentation
├── assets/                # Large files (not tracked)
└── .gitignore            # Git ignore rules
```

## Quick Start

### Prerequisites

- Python 3.8+

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Zhongbing-Chen/DocAI-UI.git
cd DocAI-UI
```

2. Install dependencies:
```bash
pip install -r src/mineru_server/requirements.txt
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
