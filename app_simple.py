#!/usr/bin/env python3
"""
Extremely simple Flask app for testing Cloud Run deployment
"""

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, ChordiSpeak is running!"

@app.route('/test')
def test():
    return jsonify({
        'status': 'ok',
        'message': 'Simple Flask app is running'
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'chordispeak-simple'
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting simple Flask app on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port) 