"""
Simple Flask Server for DocAI Frontend
Serves the frontend and provides API endpoints for document parsing.
"""

import os
import sys
import json
import uuid
import zipfile
import re
import requests
from io import BytesIO
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from mineru_client import MinerUClient, ModelVersion, TaskState
    MINERU_AVAILABLE = True
except ImportError:
    print("Warning: mineru_client not available. Some features may be limited.")
    MINERU_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend', 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend', 'results')
MAIN_RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# In-memory storage for tasks
task_store: Dict[str, Dict[str, Any]] = {}


def load_initial_tasks():
    """Load initial tasks from mineru_server/task_store_init.json and scan results directory"""
    import time

    loaded_count = 0

    # 1. Load from task_store_init.json (if exists)
    init_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'mineru_server', 'task_store_init.json')
    if os.path.exists(init_file):
        try:
            with open(init_file, 'r') as f:
                initial_tasks = json.load(f)
                # Convert relative paths to absolute paths
                for task_id, task in initial_tasks.items():
                    if 'file_path' in task and not os.path.isabs(task['file_path']):
                        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        task['file_path'] = os.path.join(project_root, task['file_path'])
                    if 'results' in task and 'folder' in task['results']:
                        if not os.path.isabs(task['results']['folder']):
                            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                            task['results']['folder'] = os.path.join(project_root, task['results']['folder'])
                task_store.update(initial_tasks)
                loaded_count = len(initial_tasks)
                print(f"✓ Loaded {loaded_count} tasks from task_store_init.json")
        except Exception as e:
            print(f"✗ Error loading task_store_init.json: {e}")

    # 2. Scan main results directory for additional tasks
    if os.path.exists(MAIN_RESULTS_FOLDER):
        scanned_count = 0
        for task_dir in os.listdir(MAIN_RESULTS_FOLDER):
            task_path = os.path.join(MAIN_RESULTS_FOLDER, task_dir)

            # Skip if already in task_store or not a directory
            if task_dir in task_store or not os.path.isdir(task_path):
                continue

            # Skip detail and table directories
            if task_dir in ['detail', 'table']:
                continue

            try:
                # Look for full.md
                md_path = os.path.join(task_path, 'full.md')
                content_list_path = os.path.join(task_path, 'content_list_v2.json')

                if not os.path.exists(md_path):
                    continue

                # Read markdown
                with open(md_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()

                # Generate coordinate mappings if content_list_v2.json exists
                coordinate_mappings = []
                if os.path.exists(content_list_path):
                    with open(content_list_path, 'r', encoding='utf-8') as f:
                        content_list = json.load(f)
                        try:
                            mappings = create_coordinate_mapping_from_content_list(content_list, markdown_content)
                            coordinate_mappings = [asdict(m) for m in mappings]
                        except Exception as e:
                            print(f"  Warning: Could not create mappings for {task_dir}: {e}")

                # Find original PDF
                pdf_file = None
                for f in os.listdir(task_path):
                    if f.endswith('_origin.pdf'):
                        pdf_file = os.path.join(task_path, f)
                        break

                # Create task entry
                task_store[task_dir] = {
                    'task_id': task_dir,
                    'data_id': task_dir,
                    'status': 'done',
                    'model_version': 'vlm',  # Assume VLM for historical tasks
                    'file_name': pdf_file.split('/')[-1] if pdf_file else f'{task_dir}.pdf',
                    'file_path': pdf_file,
                    'results': {
                        'folder': task_path,
                        'markdown': markdown_content,
                        'coordinate_mappings': coordinate_mappings,
                        'total_pages': markdown_content.count('\\n') + 1  # Rough estimate
                    }
                }
                scanned_count += 1

            except Exception as e:
                print(f"  Warning: Could not load task {task_dir}: {e}")
                continue

        if scanned_count > 0:
            print(f"✓ Scanned and loaded {scanned_count} additional tasks from results directory")

    total = len(task_store)
    print(f"✓ Total tasks in memory: {total}")

    return total


@dataclass
class CoordinateMapping:
    """Maps coordinates between original document and parsed markdown"""
    page_num: int
    bbox: List[float]  # [x0, y0, x1, y1] in normalized coordinates
    markdown_offset: int
    markdown_length: int
    element_type: str
    element_id: str


def create_coordinate_mapping_from_content_list(content_list: List[Any], markdown_content: str) -> List[CoordinateMapping]:
    """
    Extract coordinate mappings from MinerU content_list_v2.json format.

    This version uses a more robust approach:
    1. Pre-process markdown to identify element boundaries
    2. Match content_list elements to markdown positions using fuzzy matching
    3. Handle special elements (tables, images) with their markdown representations
    """
    import re

    mappings = []

    # Build a list of markdown segments with their positions
    markdown_segments = []

    # Find all tables (HTML format)
    table_pattern = r'<table>[\s\S]*?<\/table>'
    last_end = 0
    for match in re.finditer(table_pattern, markdown_content):
        # Add text before table
        if match.start() > last_end:
            text = markdown_content[last_end:match.start()]
            if text.strip():
                markdown_segments.append({
                    'type': 'text',
                    'start': last_end,
                    'end': match.start(),
                    'content': text
                })
        # Add table
        markdown_segments.append({
            'type': 'table',
            'start': match.start(),
            'end': match.end(),
            'content': match.group()
        })
        last_end = match.end()

    # Add remaining text
    if last_end < len(markdown_content):
        text = markdown_content[last_end:]
        if text.strip():
            markdown_segments.append({
                'type': 'text',
                'start': last_end,
                'end': len(markdown_content),
                'content': text
            })

    # If no segments found, treat entire markdown as one text segment
    if not markdown_segments:
        markdown_segments.append({
            'type': 'text',
            'start': 0,
            'end': len(markdown_content),
            'content': markdown_content
        })

    # Process each page and element
    segment_idx = 0
    segment_offset = 0

    for page_idx, page_content in enumerate(content_list):
        page_num = page_idx + 1

        for element in page_content:
            if not isinstance(element, dict):
                continue

            bbox = element.get('bbox', [])
            if len(bbox) != 4:
                continue

            element_type = element.get('type', 'text')

            # Extract text content from element
            element_text = ""
            content_data = element.get('content', {})

            if isinstance(content_data, dict):
                if 'title_content' in content_data:
                    for item in content_data['title_content']:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            element_text += item.get('content', '')
                elif 'paragraph_content' in content_data:
                    for item in content_data['paragraph_content']:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            element_text += item.get('content', '')
                elif 'table_content' in content_data:
                    element_type = 'table'
                    element_text = "[TABLE]"
                elif 'image_content' in content_data:
                    element_type = 'image'
                    element_text = "[IMAGE]"

            if not element_text.strip():
                continue

            # Find position in markdown
            markdown_offset = -1
            markdown_length = 0

            if element_type == 'table':
                while segment_idx < len(markdown_segments):
                    seg = markdown_segments[segment_idx]
                    if seg['type'] == 'table':
                        markdown_offset = seg['start']
                        markdown_length = seg['end'] - seg['start']
                        segment_idx += 1
                        segment_offset = 0
                        break
                    segment_idx += 1

            elif element_type == 'image':
                img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
                for match in re.finditer(img_pattern, markdown_content):
                    if match.start() >= markdown_segments[segment_idx]['start'] if segment_idx < len(markdown_segments) else 0:
                        markdown_offset = match.start()
                        markdown_length = match.end() - match.start()
                        break

            else:
                search_text = element_text.strip()
                found = False
                start_search_idx = segment_idx

                while start_search_idx < len(markdown_segments):
                    seg = markdown_segments[start_search_idx]

                    if seg['type'] == 'text':
                        seg_content = seg['content']
                        pos = seg_content.find(search_text, segment_offset if start_search_idx == segment_idx else 0)

                        if pos != -1:
                            markdown_offset = seg['start'] + pos
                            markdown_length = len(search_text)

                            if start_search_idx == segment_idx:
                                segment_offset = pos + len(search_text)
                            else:
                                segment_idx = start_search_idx
                                segment_offset = pos + len(search_text)

                            found = True
                            break

                        # Try fuzzy match
                        clean_seg = re.sub(r'[#*_`\[\]!()]+', '', seg_content)
                        clean_search = re.sub(r'[#*_`\[\]!()]+', '', search_text)

                        if clean_search in clean_seg:
                            pos = seg_content.find(search_text[:20] if len(search_text) > 20 else search_text,
                                                   segment_offset if start_search_idx == segment_idx else 0)
                            if pos != -1:
                                markdown_offset = seg['start'] + pos
                                markdown_length = min(len(search_text), len(seg_content) - pos)

                                if start_search_idx == segment_idx:
                                    segment_offset = pos + markdown_length
                                else:
                                    segment_idx = start_search_idx
                                    segment_offset = pos + markdown_length

                                found = True
                                break

                    start_search_idx += 1

                if not found:
                    if segment_idx < len(markdown_segments):
                        seg = markdown_segments[segment_idx]
                        markdown_offset = seg['start'] + segment_offset
                        markdown_length = 0

            if markdown_offset >= 0:
                if bbox[2] > 1 or bbox[3] > 1:
                    normalized_bbox = [b/1000 for b in bbox]
                else:
                    normalized_bbox = bbox

                mapping = CoordinateMapping(
                    page_num=page_num,
                    bbox=normalized_bbox,
                    markdown_offset=markdown_offset,
                    markdown_length=markdown_length if markdown_length > 0 else len(element_text),
                    element_type=element_type,
                    element_id=str(uuid.uuid4())
                )
                mappings.append(mapping)

    return mappings


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "DocAI Frontend Server"}), 200


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload a file for parsing."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    model_version_str = request.form.get('model_version', 'pipeline')
    data_id = request.form.get('data_id', str(uuid.uuid4()))

    if not MINERU_AVAILABLE:
        return jsonify({"error": "MinerU client not available. Please install mineru_client."}), 500

    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, f"{data_id}_{file.filename}")
    file.save(file_path)

    api_token = os.getenv('MINERU_API_TOKEN')
    if not api_token:
        return jsonify({"error": "Server not configured with API token"}), 500

    try:
        client = MinerUClient(api_token)

        upload_data = client.request_batch_file_upload(
            files=[{
                "name": file.filename,
                "data_id": data_id
            }],
            model_version=ModelVersion.VLM if model_version_str == 'vlm' else ModelVersion.PIPELINE
        )

        upload_url = upload_data['file_urls'][0]
        batch_id = upload_data['batch_id']

        with open(file_path, 'rb') as f:
            response = requests.put(upload_url, data=f)
            if response.status_code != 200:
                return jsonify({"error": "Failed to upload file to MinerU"}), 500

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
    """Parse a document from URL."""
    data = request.get_json()

    if not data or 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400

    url = data['url']
    model_version_str = data.get('model_version', 'pipeline')
    data_id = data.get('data_id', str(uuid.uuid4()))

    if not MINERU_AVAILABLE:
        return jsonify({"error": "MinerU client not available"}), 500

    api_token = os.getenv('MINERU_API_TOKEN')
    if not api_token:
        return jsonify({"error": "Server not configured with API token"}), 500

    try:
        client = MinerUClient(api_token)

        task_id_remote = client.create_task_from_url(
            url=url,
            model_version=ModelVersion.VLM if model_version_str == 'vlm' else ModelVersion.PIPELINE,
            data_id=data_id
        )

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
    """Get the status of a parsing task."""
    if task_id not in task_store:
        return jsonify({"error": "Task not found"}), 404

    task = task_store[task_id]
    api_token = os.getenv('MINERU_API_TOKEN')

    if not MINERU_AVAILABLE:
        return jsonify({"error": "MinerU client not available"}), 500

    try:
        client = MinerUClient(api_token)

        if 'batch_id' in task:
            batch_result = client.get_batch_results(task['batch_id'])

            if batch_result.get('extract_result'):
                result = batch_result['extract_result'][0]
                state = result.get('state', 'pending')
                task['status'] = state

                if state == 'done' and 'results' not in task:
                    zip_url = result.get('full_zip_url')
                    if zip_url:
                        task['results_url'] = zip_url
                        task['status'] = 'processing'

        elif 'remote_task_id' in task:
            remote_task_id = task['remote_task_id']
            result = client.get_task_result(remote_task_id)

            task['status'] = result.state.value

            if result.state == TaskState.DONE and 'results' not in task:
                zip_url = result.full_zip_url
                if zip_url:
                    task['results_url'] = zip_url
                    task['status'] = 'processing'

        # Process results
        if task.get('status') == 'processing' and 'results_url' in task:
            try:
                response = requests.get(task['results_url'])
                zip_file = zipfile.ZipFile(BytesIO(response.content))

                result_folder = os.path.join(RESULTS_FOLDER, task_id)
                os.makedirs(result_folder, exist_ok=True)
                zip_file.extractall(result_folder)

                json_path = os.path.join(result_folder, 'content.json')
                content_list_path = os.path.join(result_folder, 'content_list_v2.json')
                markdown_path = os.path.join(result_folder, 'full.md')
                if not os.path.exists(markdown_path):
                    markdown_path = os.path.join(result_folder, 'content.md')

                markdown_content = ""
                if os.path.exists(markdown_path):
                    with open(markdown_path, 'r', encoding='utf-8') as f:
                        markdown_content = f.read()

                coordinate_mappings = []
                json_content = None

                if os.path.exists(content_list_path):
                    with open(content_list_path, 'r', encoding='utf-8') as f:
                        content_list = json.load(f)
                        coordinate_mappings = create_coordinate_mapping_from_content_list(content_list, markdown_content)
                elif os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_content = json.load(f)

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
    """Get the results of a completed task."""
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
    """Get the original uploaded file."""
    if task_id not in task_store:
        return jsonify({"error": "Task not found"}), 404

    task = task_store[task_id]

    if 'file_path' not in task:
        return jsonify({"error": "Original file not available"}), 400

    if not os.path.exists(task['file_path']):
        return jsonify({"error": "Original file not found"}), 404

    return send_file(task['file_path'], as_attachment=True)


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
    frontend_path = os.path.join(os.path.dirname(__file__), 'index.html')
    try:
        with open(frontend_path, 'r', encoding='utf-8') as f:
            return render_template_string(f.read())
    except FileNotFoundError:
        return jsonify({
            "error": "Frontend not found",
            "message": "Please ensure frontend/index.html exists"
        }), 404


@app.route('/images/<path:filename>', methods=['GET'])
def serve_image(filename):
    """Serve image files from results directories"""
    # First, try to find the image in any result folder
    for task_id, task in task_store.items():
        if 'results' in task and 'folder' in task['results']:
            result_folder = task['results']['folder']
            # Check in images subfolder
            image_path = os.path.join(result_folder, 'images', filename)
            if os.path.exists(image_path):
                return send_file(image_path)
            # Also check root of result folder
            image_path = os.path.join(result_folder, filename)
            if os.path.exists(image_path):
                return send_file(image_path)

    # Try main results folder
    main_results = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    if os.path.exists(main_results):
        for task_dir in os.listdir(main_results):
            task_path = os.path.join(main_results, task_dir)
            if os.path.isdir(task_path):
                image_path = os.path.join(task_path, 'images', filename)
                if os.path.exists(image_path):
                    return send_file(image_path)

    return jsonify({"error": f"Image not found: {filename}"}), 404


@app.route('/<path:filename>', methods=['GET'])
def serve_static(filename):
    """Serve static files (CSS, JS)"""
    # Use the directory containing server.py as the static root
    static_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(static_dir, filename)

    print(f"DEBUG: Serving static file: {filename}")
    print(f"DEBUG: Full path: {file_path}")
    print(f"DEBUG: Exists: {os.path.exists(file_path)}")

    try:
        return send_file(file_path)
    except FileNotFoundError:
        return jsonify({"error": f"File not found: {filename}"}), 404


if __name__ == '__main__':
    # Check for API token
    if not os.getenv('MINERU_API_TOKEN'):
        print("Warning: MINERU_API_TOKEN environment variable not set")
        print("Get your token from: https://mineru.net/apiManage/token")
        print("\nYou can set it with:")
        print("  export MINERU_API_TOKEN=your_token_here")

    print("\n" + "="*60)
    print("DocAI Frontend Server")
    print("="*60)

    # Load initial tasks
    print("\n📂 Loading task history...")
    total_tasks = load_initial_tasks()
    print(f"\n✓ Ready to serve {total_tasks} tasks")

    print(f"\nFrontend URL: http://localhost:5001")
    print(f"API Base URL: http://localhost:5001/api")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5001)
