#!/usr/bin/env python3
"""
Simple Flask Server for DocAI Frontend - Clean Implementation
"""
import os
import sys
import json
from flask import Flask, request, jsonify, send_file

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__, static_folder='.')

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Load task store
task_store = {}

def load_tasks():
    """Load tasks from mineru_server task_store_init.json"""
    init_file = os.path.join(PROJECT_ROOT, 'mineru_server', 'task_store_init.json')
    if os.path.exists(init_file):
        with open(init_file, 'r') as f:
            tasks = json.load(f)
            for task_id, task in tasks.items():
                if 'file_path' in task and not os.path.isabs(task['file_path']):
                    task['file_path'] = os.path.join(PROJECT_ROOT, task['file_path'])
            task_store.update(tasks)
            print(f"✓ Loaded {len(tasks)} tasks")

load_tasks()

@app.route('/')
def index():
    """Serve the frontend"""
    return send_file(os.path.join(os.path.dirname(__file__), 'index.html'))

@app.route('/<path:path>')
def static_files(path):
    """Serve static files (CSS, JS)"""
    return send_file(os.path.join(os.path.dirname(__file__), path))

@app.route('/api/tasks', methods=['GET'])
def list_tasks():
    """List all tasks"""
    tasks = [{'task_id': tid, **task} for tid, task in task_store.items()]
    return jsonify({'tasks': tasks})

@app.route('/api/task/<task_id>/results', methods=['GET'])
def get_results(task_id):
    """Get task results"""
    if task_id not in task_store:
        return jsonify({'error': 'Task not found'}), 404

    task = task_store[task_id]
    if task.get('status') != 'done':
        return jsonify({'error': 'Task not complete'}), 400

    return jsonify({
        'markdown': task['results']['markdown'],
        'coordinate_mappings': task['results']['coordinate_mappings']
    })

@app.route('/api/task/<task_id>/original', methods=['GET'])
def get_original(task_id):
    """Get original PDF"""
    if task_id not in task_store:
        return jsonify({'error': 'Task not found'}), 404

    task = task_store[task_id]
    if 'results' not in task or 'folder' not in task['results']:
        return jsonify({'error': 'No results available'}), 400

    # Try to find PDF in results folder
    result_folder = task['results']['folder']
    pdf_files = [f for f in os.listdir(result_folder) if f.endswith('.pdf')]

    if pdf_files:
        return send_file(os.path.join(result_folder, pdf_files[0]))
    else:
        return jsonify({'error': 'PDF not found'}), 404

if __name__ == '__main__':
    print("\n" + "="*60)
    print("DocAI Frontend Server (NEW)")
    print("="*60)
    print(f"Frontend: http://localhost:5002")
    print("="*60)

    app.run(host='0.0.0.0', port=5002, debug=True)
