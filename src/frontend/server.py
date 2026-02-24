#!/usr/bin/env python3
"""
Simple Flask Server for DocAI Frontend - Clean Implementation
"""
import os
import sys
import json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__, static_folder='.')
CORS(app)  # Enable CORS for all routes

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Load task store
task_store = {}

def load_tasks():
    """Load tasks from mineru_server task_store_init.json and scan results directory"""
    init_file = os.path.join(PROJECT_ROOT, 'mineru_server', 'task_store_init.json')

    # 1. Load from task_store_init.json
    if os.path.exists(init_file):
        try:
            with open(init_file, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
                for task_id, task in tasks.items():
                    # Convert relative paths to absolute paths
                    if 'file_path' in task and not os.path.isabs(task['file_path']):
                        task['file_path'] = os.path.join(PROJECT_ROOT, task['file_path'])
                    if 'results' in task and 'folder' in task['results']:
                        if not os.path.isabs(task['results']['folder']):
                            task['results']['folder'] = os.path.join(PROJECT_ROOT, task['results']['folder'])
                task_store.update(tasks)
                print(f"✓ Loaded {len(tasks)} tasks from task_store_init.json")
        except Exception as e:
            print(f"✗ Error loading task_store_init.json: {e}")

    # 2. Scan main results directory for additional tasks
    main_results_folder = os.path.join(PROJECT_ROOT, 'results')
    if os.path.exists(main_results_folder):
        scanned_count = 0
        for task_dir in os.listdir(main_results_folder):
            task_path = os.path.join(main_results_folder, task_dir)

            # Skip if already in task_store or not a directory
            if task_dir in task_store or not os.path.isdir(task_path):
                continue

            # Skip non-task directories
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
                    try:
                        with open(content_list_path, 'r', encoding='utf-8') as f:
                            content_list = json.load(f)

                        import uuid
                        for page_idx, page_content in enumerate(content_list):
                            if not isinstance(page_content, list):
                                continue

                            page_num = page_idx + 1
                            for element in page_content:
                                if not isinstance(element, dict):
                                    continue

                                bbox = element.get('bbox', [])
                                if len(bbox) != 4:
                                    continue

                                element_type = element.get('type', 'text')

                                # Calculate text content length for markdown offset estimation
                                text_content = ""
                                if 'content' in element:
                                    content = element['content']
                                    if isinstance(content, dict):
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

                                # Find position in markdown
                                markdown_offset = markdown_content.find(text_content) if text_content else 0
                                if markdown_offset == -1:
                                    markdown_offset = 0

                                mapping = {
                                    'page_num': page_num,
                                    'bbox': [bbox[0]/1000, bbox[1]/1000, bbox[2]/1000, bbox[3]/1000],  # Normalize
                                    'markdown_offset': markdown_offset,
                                    'markdown_length': len(text_content),
                                    'element_type': element_type,
                                    'element_id': str(uuid.uuid4())
                                }
                                coordinate_mappings.append(mapping)
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
                        'total_pages': markdown_content.count('\n') + 1  # Rough estimate
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

# Load tasks on startup
print("\n" + "="*60)
print("DocAI Frontend Server")
print("="*60)
print("\n📂 Loading task history...")
load_tasks()
print(f"\n✓ Ready to serve {len(task_store)} tasks")
print(f"\nFrontend URL: http://localhost:5002")
print(f"API Base URL: http://localhost:5002/api")
print("="*60 + "\n")

@app.route('/')
def index():
    """Serve the frontend"""
    return send_file(os.path.join(os.path.dirname(__file__), 'index.html'))

@app.route('/<path:path>')
def static_files(path):
    """Serve static files (CSS, JS)"""
    file_path = os.path.join(os.path.dirname(__file__), path)
    if os.path.exists(file_path):
        return send_file(file_path)
    return jsonify({'error': f'File not found: {path}'}), 404

@app.route('/api/tasks', methods=['GET'])
def list_tasks():
    """List all tasks"""
    tasks = []
    for tid, task in task_store.items():
        tasks.append({
            'task_id': tid,
            'file_name': task.get('file_name'),
            'status': task.get('status'),
            'model_version': task.get('model_version')
        })
    return jsonify({'tasks': tasks})

@app.route('/api/task/<task_id>/results', methods=['GET'])
def get_results(task_id):
    """Get task results"""
    if task_id not in task_store:
        return jsonify({'error': 'Task not found'}), 404

    task = task_store[task_id]
    if task.get('status') != 'done':
        return jsonify({'error': 'Task not complete'}), 400

    if 'results' not in task:
        return jsonify({'error': 'No results available'}), 400

    return jsonify({
        'markdown': task['results'].get('markdown', ''),
        'coordinate_mappings': task['results'].get('coordinate_mappings', [])
    })

@app.route('/api/task/<task_id>/original', methods=['GET'])
def get_original(task_id):
    """Get original PDF"""
    if task_id not in task_store:
        return jsonify({'error': 'Task not found'}), 404

    task = task_store[task_id]

    # Try to get from results folder first
    if 'results' in task and 'folder' in task['results']:
        result_folder = task['results']['folder']
        if os.path.exists(result_folder):
            pdf_files = [f for f in os.listdir(result_folder) if f.endswith('.pdf')]
            if pdf_files:
                return send_file(os.path.join(result_folder, pdf_files[0]))

    # Try file_path
    if 'file_path' in task and os.path.exists(task['file_path']):
        return send_file(task['file_path'])

    return jsonify({'error': 'PDF not found'}), 404

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'tasks_loaded': len(task_store)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
