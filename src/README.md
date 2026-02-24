# DocAI Source Code

This directory contains the main source code for DocAI.

## Structure

```
src/
├── mineru_client/     # MinerU API client library
├── mineru_server/     # Flask server for MinerU API
└── frontend/          # Web frontend UI
```

## Quick Start

### Frontend Server

```bash
cd frontend
python server.py
```

Then open http://localhost:5002 in your browser.

### MinerU Server

```bash
cd mineru_server
pip install -r requirements.txt
python app.py
```

## Requirements

- Python 3.8+
- Flask
- See `mineru_server/requirements.txt` for full dependencies
