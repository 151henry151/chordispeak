# ChordiSpeak - AI Chord Vocal Generator

**Version:** 1.0.0

ChordiSpeak generates a new vocal track for your song that speaks the chord names in the original singer's voice, perfectly timed to the music.

## Features
- **Vocal/Instrumental Separation**: Uses Demucs AI to split uploaded audio into vocals and instrumental.
- **Chord Detection**: Analyzes the instrumental track to detect chord progressions with precise timing.
- **Voice Cloning**: Uses the full vocal track with Coqui TTS XTTS v2 to synthesize spoken chord names in the original singer's voice.
- **Audio Mixing**: Overlays the synthesized chord vocals onto the instrumental track.
- **Simple Web Interface**: Upload a file and download the result.

## How It Works
1. **Upload** your audio file (MP3, WAV, FLAC, M4A) via the web interface.
2. **Separation**: Demucs splits the file into `vocal_track.wav` and `instrumental_track.wav`.
3. **Voice Sample Extraction**: The full vocal track is used for voice cloning to capture the complete voice characteristics.
4. **Chord Detection**: Chords are detected from the instrumental using onset detection and beat tracking for precise timing.
5. **Synthesis**: Coqui TTS XTTS v2 generates spoken chord names using the full vocal track.
6. **Mixing**: The synthesized vocals are overlaid onto the instrumental.
7. **Download** your new track with chord vocals.

## Requirements
- Python 3.11
- FFmpeg (for audio processing)

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd chordispeak
   ```
2. Create and activate a Python 3.11 virtual environment:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the server:
   ```bash
   ./start.sh
   ```
5. Open `index.html` in your browser.

## API Endpoints
- `POST /upload` — Upload an audio file for processing
- `GET /status/<task_id>` — Check processing status
- `GET /download/<task_id>` — Download the processed audio
- `GET /health` — Health check

## Project Structure
```
chordispeak/
├── app.py           # Flask backend API
├── run.py           # Dev server launcher
├── index.html       # Web interface
├── requirements.txt # Python dependencies
├── version.py       # Version management script
├── VERSION          # Current version file
├── uploads/         # Uploaded and processed files
└── README.md        # This file
```

## Version Management

ChordiSpeak uses semantic versioning (MAJOR.MINOR.PATCH). Use the version management script:

```bash
# Check current version
python version.py current

# Bump versions
python version.py bump-patch  # 1.0.0 -> 1.0.1
python version.py bump-minor  # 1.0.0 -> 1.1.0  
python version.py bump-major  # 1.0.0 -> 2.0.0

# Set specific version
python version.py set 1.2.3
```

## License
This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
