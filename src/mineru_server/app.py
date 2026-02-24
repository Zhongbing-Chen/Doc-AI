"""
Flask Backend Server for MinerU Document Comparison Viewer
Handles file uploads, parsing coordination, and serves results with coordinate mapping.
"""

import os
import json
import uuid
import zipfile
import requests
from io import BytesIO
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import sys

# Add parent directory to path to import mineru_client
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mineru_client import MinerUClient, ModelVersion, TaskState

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# In-memory storage for tasks (use Redis or database in production)
task_store: Dict[str, Dict[str, Any]] = {}

# Load initial task data if available
def load_initial_tasks():
    """Load initial tasks from JSON file if exists"""
    init_file = os.path.join(os.path.dirname(__file__), 'task_store_init.json')
    if os.path.exists(init_file):
        try:
            with open(init_file, 'r') as f:
                initial_tasks = json.load(f)
                # Convert relative paths to absolute paths
                for task_id, task in initial_tasks.items():
                    if 'file_path' in task and not os.path.isabs(task['file_path']):
                        # Make path relative to project root (parent of mineru_server)
                        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        task['file_path'] = os.path.join(project_root, task['file_path'])
                task_store.update(initial_tasks)
                print(f"Loaded {len(initial_tasks)} initial tasks into task store")
        except Exception as e:
            print(f"Error loading initial tasks: {e}")

# Load initial tasks on startup
load_initial_tasks()


@dataclass
class CoordinateMapping:
    """Maps coordinates between original document and parsed markdown"""
    page_num: int
    bbox: List[float]  # [x0, y0, x1, y1] in PDF coordinates
    markdown_offset: int  # Character offset in markdown
    markdown_length: int  # Length of text in markdown
    element_type: str  # 'text', 'table', 'image', 'formula'
    element_id: str  # Unique identifier


def create_coordinate_mapping_from_content_list(content_list: List[Any], markdown_content: str) -> List[CoordinateMapping]:
    """
    Extract coordinate mappings from MinerU content_list_v2.json format.

    Args:
        content_list: Parsed content list from MinerU (list of pages)
        markdown_content: Full markdown content for offset calculation

    Returns:
        List of CoordinateMapping objects
    """
    mappings = []
    current_offset = 0

    # Process each page
    for page_idx, page_content in enumerate(content_list):
        page_num = page_idx + 1  # Pages are 1-indexed

        # Process each element on the page
        for element in page_content:
            if not isinstance(element, dict):
                continue

            bbox = element.get('bbox', [])
            if len(bbox) != 4:
                continue

            # Calculate markdown offset and length
            element_type = element.get('type', 'text')

            # Extract text content for offset calculation
            text_content = ""
            if 'content' in element:
                content = element['content']
                if isinstance(content, dict):
                    # Extract text from various content types
                    if 'title_content' in content:
                        for item in content['title_content']:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                text_content += item.get('content', '')
                    elif 'paragraph_content' in content:
                        for item in content['paragraph_content']:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                text_content += item.get('content', '')
                    elif 'table_content' in content:
                        text_content = "[TABLE]"
                    elif 'image_content' in content:
                        text_content = "[IMAGE]"

            # Find this text in the markdown
            markdown_offset = markdown_content.find(text_content, current_offset)
            if markdown_offset == -1:
                markdown_offset = current_offset

            markdown_length = len(text_content)

            mapping = CoordinateMapping(
                page_num=page_num,
                bbox=[bbox[0]/1000, bbox[1]/1000, bbox[2]/1000, bbox[3]/1000],  # Convert to normalized coordinates
                markdown_offset=markdown_offset,
                markdown_length=markdown_length,
                element_type=element_type,
                element_id=str(uuid.uuid4())
            )
            mappings.append(mapping)

            # Update current offset
            current_offset = markdown_offset + markdown_length

    return mappings


def create_coordinate_mapping(json_result: Dict[str, Any]) -> List[CoordinateMapping]:
    """
    Extract coordinate mappings from MinerU JSON output.

    Args:
        json_result: Parsed JSON result from MinerU

    Returns:
        List of CoordinateMapping objects
    """
    mappings = []

    # Process each page
    for page in json_result.get('pages', []):
        page_num = page.get('page_no', 0)

        # Process layout items
        for layout in page.get('layouts', []):
            bbox = layout.get('bbox', [])
            if len(bbox) == 4:
                mapping = CoordinateMapping(
                    page_num=page_num,
                    bbox=bbox,
                    markdown_offset=layout.get('markdown_offset', 0),
                    markdown_length=layout.get('markdown_length', 0),
                    element_type=layout.get('type', 'text'),
                    element_id=layout.get('id', str(uuid.uuid4()))
                )
                mappings.append(mapping)

        # Process tables
        for table in page.get('tables', []):
            bbox = table.get('bbox', [])
            if len(bbox) == 4:
                mapping = CoordinateMapping(
                    page_num=page_num,
                    bbox=bbox,
                    markdown_offset=table.get('markdown_offset', 0),
                    markdown_length=table.get('markdown_length', 0),
                    element_type='table',
                    element_id=table.get('id', str(uuid.uuid4()))
                )
                mappings.append(mapping)

        # Process images
        for image in page.get('images', []):
            bbox = image.get('bbox', [])
            if len(bbox) == 4:
                mapping = CoordinateMapping(
                    page_num=page_num,
                    bbox=bbox,
                    markdown_offset=image.get('markdown_offset', 0),
                    markdown_length=image.get('markdown_length', 0),
                    element_type='image',
                    element_id=image.get('id', str(uuid.uuid4()))
                )
                mappings.append(mapping)

    return mappings


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "MinerU Server"}), 200


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Upload a file for parsing.

    Request:
        - file: File to upload
        - model_version: pipeline, vlm, or MinerU-HTML (default: pipeline)
        - data_id: Optional custom data ID
        - parse_options: Optional JSON string with parsing options

    Returns:
        JSON with task_id for tracking
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Get parameters
    model_version_str = request.form.get('model_version', 'pipeline')
    data_id = request.form.get('data_id', str(uuid.uuid4()))
    parse_options_json = request.form.get('parse_options', '{}')

    try:
        parse_options = json.loads(parse_options_json)
    except json.JSONDecodeError:
        parse_options = {}

    # Map model version string to enum
    model_version_map = {
        'pipeline': ModelVersion.PIPELINE,
        'vlm': ModelVersion.VLM,
        'mineru-html': ModelVersion.MINERU_HTML
    }
    model_version = model_version_map.get(model_version_str.lower(), ModelVersion.PIPELINE)

    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, f"{data_id}_{file.filename}")
    file.save(file_path)

    # Get API token from environment
    api_token = os.getenv('MINERU_API_TOKEN')
    if not api_token:
        return jsonify({"error": "Server not configured with API token"}), 500

    try:
        # Request upload URL from MinerU
        client = MinerUClient(api_token)

        # For URL-based upload, we'd need to host the file publicly
        # For now, we'll use the batch upload API
        upload_data = client.request_batch_file_upload(
            files=[{
                "name": file.filename,
                "data_id": data_id,
                **parse_options
            }],
            model_version=model_version
        )

        # Upload the file
        upload_url = upload_data['file_urls'][0]
        batch_id = upload_data['batch_id']

        with open(file_path, 'rb') as f:
            response = requests.put(upload_url, data=f)
            if response.status_code != 200:
                return jsonify({"error": "Failed to upload file to MinerU"}), 500

        # Store task info
        task_id = str(uuid.uuid4())
        task_store[task_id] = {
            "data_id": data_id,
            "batch_id": batch_id,
            "file_name": file.filename,
            "file_path": file_path,
            "status": "pending",
            "model_version": model_version_str
        }

        return jsonify({
            "task_id": task_id,
            "data_id": data_id,
            "batch_id": batch_id,
            "status": "pending"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/parse/url', methods=['POST'])
def parse_from_url():
    """
    Parse a document from URL.

    Request:
        - url: URL of the document
        - model_version: pipeline, vlm, or MinerU-HTML (default: pipeline)
        - data_id: Optional custom data ID
        - parse_options: Optional JSON with parsing options

    Returns:
        JSON with task_id for tracking
    """
    data = request.get_json()

    if not data or 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400

    url = data['url']
    model_version_str = data.get('model_version', 'pipeline')
    data_id = data.get('data_id', str(uuid.uuid4()))
    parse_options = data.get('parse_options', {})

    # Map model version
    model_version_map = {
        'pipeline': ModelVersion.PIPELINE,
        'vlm': ModelVersion.VLM,
        'mineru-html': ModelVersion.MINERU_HTML
    }
    model_version = model_version_map.get(model_version_str.lower(), ModelVersion.PIPELINE)

    # Get API token
    api_token = os.getenv('MINERU_API_TOKEN')
    if not api_token:
        return jsonify({"error": "Server not configured with API token"}), 500

    try:
        client = MinerUClient(api_token)

        task_id_remote = client.create_task_from_url(
            url=url,
            model_version=model_version,
            data_id=data_id,
            **parse_options
        )

        # Store task info
        task_id = str(uuid.uuid4())
        task_store[task_id] = {
            "data_id": data_id,
            "remote_task_id": task_id_remote,
            "url": url,
            "status": "pending",
            "model_version": model_version_str
        }

        return jsonify({
            "task_id": task_id,
            "remote_task_id": task_id_remote,
            "data_id": data_id,
            "status": "pending"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/task/<task_id>', methods=['GET'])
def get_task_status(task_id: str):
    """
    Get the status of a parsing task.

    Returns:
        JSON with task status and results if complete
    """
    if task_id not in task_store:
        return jsonify({"error": "Task not found"}), 404

    task = task_store[task_id]
    api_token = os.getenv('MINERU_API_TOKEN')

    try:
        client = MinerUClient(api_token)

        # Check status based on type of task
        if 'batch_id' in task:
            # Batch upload task
            batch_result = client.get_batch_results(task['batch_id'])

            if batch_result.get('extract_result'):
                result = batch_result['extract_result'][0]
                state = result.get('state', 'pending')

                task['status'] = state

                if state == 'done' and 'results' not in task:
                    # Download and process results
                    zip_url = result.get('full_zip_url')
                    if zip_url:
                        task['results_url'] = zip_url
                        task['status'] = 'processing'

        elif 'remote_task_id' in task:
            # URL-based task
            remote_task_id = task['remote_task_id']
            result = client.get_task_result(remote_task_id)

            task['status'] = result.state.value

            if result.state == TaskState.DONE and 'results' not in task:
                zip_url = result.full_zip_url
                if zip_url:
                    task['results_url'] = zip_url
                    task['status'] = 'processing'

        # If we have results URL but haven't processed yet
        if task.get('status') == 'processing' and 'results_url' in task:
            try:
                # Download results
                response = requests.get(task['results_url'])
                zip_file = zipfile.ZipFile(BytesIO(response.content))

                # Extract files
                result_folder = os.path.join(RESULTS_FOLDER, task_id)
                os.makedirs(result_folder, exist_ok=True)
                zip_file.extractall(result_folder)

                # Parse JSON for coordinate mapping
                json_path = os.path.join(result_folder, 'content.json')
                content_list_path = os.path.join(result_folder, 'content_list_v2.json')
                # Try full.md first, fall back to content.md
                markdown_path = os.path.join(result_folder, 'full.md')
                if not os.path.exists(markdown_path):
                    markdown_path = os.path.join(result_folder, 'content.md')

                # Read markdown first
                markdown_content = ""
                if os.path.exists(markdown_path):
                    with open(markdown_path, 'r', encoding='utf-8') as f:
                        markdown_content = f.read()

                coordinate_mappings = []
                json_content = None

                # Try content_list_v2.json first (newer format)
                if os.path.exists(content_list_path):
                    with open(content_list_path, 'r', encoding='utf-8') as f:
                        content_list = json.load(f)
                        coordinate_mappings = create_coordinate_mapping_from_content_list(content_list, markdown_content)
                        print(f"Created {len(coordinate_mappings)} coordinate mappings from content_list_v2.json")
                # Fall back to content.json (older format)
                elif os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_content = json.load(f)
                        coordinate_mappings = create_coordinate_mapping(json_content)
                        print(f"Created {len(coordinate_mappings)} coordinate mappings from content.json")

                task['results'] = {
                    'folder': result_folder,
                    'markdown': markdown_content,
                    'coordinate_mappings': [asdict(m) for m in coordinate_mappings],
                    'json_data': json_content
                }
                task['status'] = 'done'

            except Exception as e:
                task['status'] = 'error'
                task['error'] = str(e)

        return jsonify({
            "task_id": task_id,
            "data_id": task.get('data_id'),
            "status": task['status'],
            "model_version": task.get('model_version'),
            "results": task.get('results')
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/task/<task_id>/results', methods=['GET'])
def get_task_results(task_id: str):
    """
    Get the results of a completed task.

    Returns:
        JSON with markdown content and coordinate mappings
    """
    if task_id not in task_store:
        return jsonify({"error": "Task not found"}), 404

    task = task_store[task_id]

    if task.get('status') != 'done' or 'results' not in task:
        return jsonify({"error": "Task not complete or no results available"}), 400

    return jsonify({
        "task_id": task_id,
        "markdown": task['results']['markdown'],
        "coordinate_mappings": task['results']['coordinate_mappings']
    }), 200


@app.route('/api/task/<task_id>/original', methods=['GET'])
def get_original_file(task_id: str):
    """
    Get the original uploaded file.

    Returns:
        File download
    """
    if task_id not in task_store:
        return jsonify({"error": "Task not found"}), 404

    task = task_store[task_id]

    if 'file_path' not in task:
        return jsonify({"error": "Original file not available"}), 400

    if not os.path.exists(task['file_path']):
        return jsonify({"error": "Original file not found"}), 404

    return send_file(task['file_path'], as_attachment=True)


@app.route('/api/task/<task_id>/coordinate', methods=['POST'])
def find_coordinate(task_id: str):
    """
    Find coordinate mapping based on query.

    Request:
        - query_type: 'markdown_to_original' or 'original_to_markdown'
        - markdown_offset: Character offset in markdown (for markdown_to_original)
        - page_num: Page number (for original_to_markdown)
        - bbox: Bounding box [x0, y0, x1, y1] (for original_to_markdown)

    Returns:
        JSON with matching coordinates
    """
    if task_id not in task_store:
        return jsonify({"error": "Task not found"}), 404

    task = task_store[task_id]

    if task.get('status') != 'done' or 'results' not in task:
        return jsonify({"error": "Task not complete or no results available"}), 400

    data = request.get_json()
    query_type = data.get('query_type')
    mappings = task['results']['coordinate_mappings']

    if query_type == 'markdown_to_original':
        markdown_offset = data.get('markdown_offset')
        if markdown_offset is None:
            return jsonify({"error": "markdown_offset required"}), 400

        # Find mapping that contains this offset
        for mapping in mappings:
            if (mapping['markdown_offset'] <= markdown_offset <
                mapping['markdown_offset'] + mapping['markdown_length']):
                return jsonify({
                    "page_num": mapping['page_num'],
                    "bbox": mapping['bbox'],
                    "element_type": mapping['element_type'],
                    "element_id": mapping['element_id']
                }), 200

        return jsonify({"error": "No mapping found"}), 404

    elif query_type == 'original_to_markdown':
        page_num = data.get('page_num')
        bbox = data.get('bbox')

        if page_num is None or bbox is None:
            return jsonify({"error": "page_num and bbox required"}), 400

        # Find mapping that intersects with this bbox
        x0, y0, x1, y1 = bbox

        for mapping in mappings:
            if mapping['page_num'] != page_num:
                continue

            mx0, my0, mx1, my1 = mapping['bbox']

            # Check intersection
            if not (x1 < mx0 or x0 > mx1 or y1 < my0 or y0 > my1):
                return jsonify({
                    "markdown_offset": mapping['markdown_offset'],
                    "markdown_length": mapping['markdown_length'],
                    "element_type": mapping['element_type'],
                    "element_id": mapping['element_id']
                }), 200

        return jsonify({"error": "No mapping found"}), 404

    else:
        return jsonify({"error": "Invalid query_type"}), 400


@app.route('/api/tasks', methods=['GET'])
def list_tasks():
    """List all tasks"""
    tasks_summary = []
    for task_id, task in task_store.items():
        tasks_summary.append({
            "task_id": task_id,
            "data_id": task.get('data_id'),
            "status": task.get('status'),
            "model_version": task.get('model_version'),
            "file_name": task.get('file_name'),
            "url": task.get('url')
        })

    return jsonify({"tasks": tasks_summary}), 200


@app.route('/api/task/<task_id>', methods=['DELETE'])
def delete_task(task_id: str):
    """Delete a task and its results"""
    if task_id not in task_store:
        return jsonify({"error": "Task not found"}), 404

    task = task_store[task_id]

    # Clean up files
    if 'file_path' in task and os.path.exists(task['file_path']):
        os.remove(task['file_path'])

    if 'results' in task and 'folder' in task['results']:
        import shutil
        if os.path.exists(task['results']['folder']):
            shutil.rmtree(task['results']['folder'])

    del task_store[task_id]

    return jsonify({"message": "Task deleted"}), 200


@app.route('/', methods=['GET'])
def index():
    """Serve the frontend"""
    frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mineru_frontend', 'index.html')
    try:
        with open(frontend_path, 'r', encoding='utf-8') as f:
            return render_template_string(f.read())
    except FileNotFoundError:
        return jsonify({
            "error": "Frontend not found",
            "message": "Please ensure mineru_frontend/index.html exists"
        }), 404


@app.route('/<path:filename>', methods=['GET'])
def serve_static(filename):
    """Serve static files (CSS, JS)"""
    static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mineru_frontend')
    try:
        return send_file(os.path.join(static_path, filename))
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404


if __name__ == '__main__':
    # Check for API token
    if not os.getenv('MINERU_API_TOKEN'):
        print("Warning: MINERU_API_TOKEN environment variable not set")
        print("Get your token from: https://mineru.net/apiManage/token")

    app.run(debug=True, host='0.0.0.0', port=5000)
