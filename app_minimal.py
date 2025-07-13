#!/usr/bin/env python3
"""
ChordiSpeak - AI Chord Vocal Generator (Minimal Version)
Copyright (C) 2024

This is a minimal version for testing Cloud Run deployment.
"""

import os
import uuid
import json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

def get_version():
    """Read version from VERSION file"""
    try:
        with open('VERSION', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return "1.0.0"  # fallback version

VERSION = get_version()

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Task storage (in production, use Redis or database)
tasks = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main application page"""
    try:
        return send_file('index.html')
    except Exception as e:
        print(f"Error serving index.html: {e}")
        return f"Error: {str(e)}", 500

@app.route('/test')
def test():
    """Simple test route to verify the app is working"""
    return jsonify({
        'status': 'ok',
        'message': 'Flask app is running (minimal version)',
        'version': VERSION
    })

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': VERSION,
        'name': 'ChordiSpeak (Minimal)'
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Minimal upload endpoint that just returns a message"""
    return jsonify({
        'error': 'Upload functionality not available in minimal version',
        'message': 'This is a minimal version for testing deployment'
    }), 501

@app.route('/status/<task_id>')
def get_status(task_id):
    return jsonify({
        'error': 'Status functionality not available in minimal version',
        'message': 'This is a minimal version for testing deployment'
    }), 501

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting minimal Flask app on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port) 