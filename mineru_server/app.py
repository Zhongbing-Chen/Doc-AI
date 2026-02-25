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

    This version uses a more robust approach:
    1. Pre-process markdown to identify element boundaries
    2. Match content_list elements to markdown positions using fuzzy matching
    3. Handle special elements (tables, images) with their markdown representations
    """
    import re

    mappings = []

    # Build a list of markdown segments with their positions
    # Each segment: {type: 'text'|'table'|'image', start: int, end: int, content: str}
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
    segment_offset = 0  # Offset within current segment

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

            # Skip empty elements
            if not element_text.strip():
                continue

            # Find position in markdown
            markdown_offset = -1
            markdown_length = 0

            if element_type == 'table':
                # Find next table in segments
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
                # Find next image markdown ![...](...)
                img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
                for match in re.finditer(img_pattern, markdown_content):
                    if match.start() >= markdown_segments[segment_idx]['start'] if segment_idx < len(markdown_segments) else 0:
                        markdown_offset = match.start()
                        markdown_length = match.end() - match.start()
                        break

            else:
                # Text element - search in text segments
                # Clean text for matching (remove extra whitespace)
                search_text = element_text.strip()

                # Try to find in current and subsequent segments
                found = False
                start_search_idx = segment_idx

                while start_search_idx < len(markdown_segments):
                    seg = markdown_segments[start_search_idx]

                    if seg['type'] == 'text':
                        # Search within this segment
                        seg_content = seg['content']
                        search_start = segment_offset if start_search_idx == segment_idx else 0

                        # Try exact match first
                        pos = seg_content.find(search_text, search_start)

                        if pos != -1:
                            markdown_offset = seg['start'] + pos
                            markdown_length = len(search_text)

                            # Update position for next search
                            if start_search_idx == segment_idx:
                                segment_offset = pos + len(search_text)
                            else:
                                segment_idx = start_search_idx
                                segment_offset = pos + len(search_text)

                            found = True
                            break

                        # Try line-by-line matching for multi-line text
                        seg_lines = seg_content.split('\n')
                        current_pos = seg['start']
                        for line in seg_lines:
                            line_stripped = line.strip()
                            if line_stripped and search_text in line_stripped:
                                # Found in this line
                                line_pos = line.find(search_text[:min(20, len(search_text))])
                                if line_pos == -1:
                                    line_pos = 0
                                markdown_offset = current_pos + line_pos
                                markdown_length = min(len(search_text), len(line) - line_pos)

                                # Update position
                                line_end_pos = seg_content.find('\n', search_start)
                                if line_end_pos == -1:
                                    line_end_pos = len(seg_content)
                                if start_search_idx == segment_idx:
                                    segment_offset = line_end_pos + 1
                                else:
                                    segment_idx = start_search_idx
                                    segment_offset = line_end_pos + 1

                                found = True
                                break
                            current_pos += len(line) + 1  # +1 for newline

                        if found:
                            break

                        # Try fuzzy match (remove markdown markers and LaTeX)
                        # Remove common markdown and LaTeX patterns
                        clean_seg = re.sub(r'[#*_`\[\]!(){}|$^+=\\/\-\.\d]', '', seg_content)
                        clean_search = re.sub(r'[#*_`\[\]!(){}|$^+=\\/\-\.\d]', '', search_text)

                        # Also remove whitespace for comparison
                        clean_seg_no_space = re.sub(r'\s', '', clean_seg)
                        clean_search_no_space = re.sub(r'\s', '', clean_search)

                        if clean_search_no_space and clean_search_no_space in clean_seg_no_space:
                            # Find approximate position by matching first few characters
                            search_prefix = clean_search[:min(10, len(clean_search))]
                            if search_prefix:
                                # Find where this prefix appears in original
                                for i in range(search_start, len(seg_content) - len(search_prefix) + 1):
                                    test_clean = re.sub(r'[#*_`\[\]!(){}|$^+=\\/\-\.\d\s]', '', seg_content[i:i+len(search_prefix)*2])
                                    if search_prefix in test_clean:
                                        markdown_offset = seg['start'] + i
                                        # Estimate length based on character ratio
                                        if clean_search_no_space:
                                            ratio = len(search_text) / len(clean_search_no_space)
                                            est_clean_len = len(clean_search_no_space)
                                            markdown_length = min(int(est_clean_len * ratio * 1.5), len(seg_content) - i)
                                        else:
                                            markdown_length = len(search_text)

                                        if start_search_idx == segment_idx:
                                            segment_offset = i + markdown_length
                                        else:
                                            segment_idx = start_search_idx
                                            segment_offset = i + markdown_length

                                        found = True
                                        break

                        if found:
                            break

                    start_search_idx += 1

                if not found:
                    # Fallback: use current position
                    if segment_idx < len(markdown_segments):
                        seg = markdown_segments[segment_idx]
                        markdown_offset = seg['start'] + segment_offset
                        markdown_length = 0  # Unknown length

                            # Update position for next search
                            if start_search_idx == segment_idx:
                                segment_offset = pos + len(search_text)
                            else:
                                segment_idx = start_search_idx
                                segment_offset = pos + len(search_text)

                            found = True
                            break

                        # Try fuzzy match (remove markdown markers)
                        # Remove common markdown patterns for comparison
                        clean_seg = re.sub(r'[#*_`\[\]!()]+', '', seg_content)
                        clean_search = re.sub(r'[#*_`\[\]!()]+', '', search_text)

                        if clean_search in clean_seg:
                            # Find approximate position
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
                    # Fallback: use current position
                    if segment_idx < len(markdown_segments):
                        seg = markdown_segments[segment_idx]
                        markdown_offset = seg['start'] + segment_offset
                        markdown_length = 0  # Unknown length

            # Only create mapping if we found a valid position
            if markdown_offset >= 0:
                # Normalize bbox (handle both 0-1000 and 0-1 ranges)
                if bbox[2] > 1 or bbox[3] > 1:  # Likely 0-1000 range
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
