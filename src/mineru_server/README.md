# MinerU Server

Flask backend server for the MinerU Document Comparison Viewer.

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install mineru_client
cd ..
pip install -e .
cd mineru_server
```

## Configuration

Set your MinerU API token:

```bash
export MINERU_API_TOKEN="your-api-token-here"
```

Get your token from: https://mineru.net/apiManage/token

## Running

```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

See [README_MINERU.md](../README_MINERU.md) for full API documentation.

## Development

The server serves the frontend from the root path `/` and provides REST API endpoints under `/api/`.

### File Structure

```
uploads/          # Uploaded files (temporary)
results/          # Parsed results (temporary)
app.py            # Flask application
requirements.txt  # Python dependencies
```

### Environment Variables

- `MINERU_API_TOKEN`: Your MinerU API token (required)
- `FLASK_ENV`: Set to `development` for debug mode (optional)
- `FLASK_PORT`: Port to run the server on (default: 5000) (optional)
