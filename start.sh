#!/bin/bash

# ChordiSpeak - AI Chord Vocal Generator
# Copyright (C) 2024
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# ChordiSpeak Startup Script
echo "🎵 Starting ChordiSpeak..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run: python -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if required packages are installed
echo "📦 Checking dependencies..."
if ! python -c "import flask" 2>/dev/null; then
    echo "❌ Flask not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Start the server
echo "🚀 Launching ChordiSpeak server..."
echo "📝 HTTP requests will be logged to chordispeak.log"

# Create log directory if it doesn't exist
mkdir -p logs

# Run the server with output filtering
python run.py 2>&1 | while IFS= read -r line; do
    if [[ "$line" == *"GET /status"* ]]; then
        echo "$line" >> logs/chordispeak.log
    else
        echo "$line"
        echo "$line" >> logs/chordispeak.log
    fi
done 