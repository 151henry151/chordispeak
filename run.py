#!/usr/bin/env python3
"""
ChordiSpeak - AI Chord Vocal Generator
Copyright (C) 2024

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

ChordiSpeak Development Server
Run this script to start the backend API server.
"""

import os
import sys
from app import app, VERSION

def main():
    print("🎵 Starting ChordiSpeak Backend Server...")
    print(f"📦 Version: {VERSION}")
    print("📡 API will be available at: http://localhost:5001")
    print("🌐 Open index.html in your browser to access the web interface")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Disable Flask's default request logging to reduce terminal noise
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    # Ensure uploads directory exists
    os.makedirs('uploads', exist_ok=True)
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5001)
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
        sys.exit(0)

if __name__ == '__main__':
    main()
