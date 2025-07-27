# ChordiSpeak - AI Chord Vocal Generator

**Version:** 1.0.0

ChordiSpeak generates a new vocal track for your song that speaks the chord names in the original singer's voice, perfectly timed to the music.

**Live Demo**: https://chordispeak.com

## Features
- **Vocal/Instrumental Separation**: Uses Demucs AI (htdemucs model) to split uploaded audio into vocals and instrumental
- **Advanced Chord Detection**: Uses Madmom's deep learning-based chord recognition with robust error handling
- **Voice Cloning**: Uses the full vocal track with Coqui TTS XTTS v2 to synthesize spoken chord names in the original singer's voice
- **Audio Mixing**: Overlays the synthesized chord vocals onto the instrumental track with precise timing
- **Real-time Progress Tracking**: Detailed progress updates with demucs percentage tracking
- **Simple Web Interface**: Upload a file and download the result
- **Robust Error Handling**: Multiple fallback mechanisms for reliable processing

## How It Works
1. **Upload** your audio file (MP3, WAV, FLAC, M4A) via the web interface (max 50MB)
2. **Audio Preparation**: Converts to WAV format using pydub
3. **Vocal Separation**: Demucs splits the file into `vocal_track.wav` and `instrumental_track.wav` (10-15 minutes)
4. **Voice Sample Extraction**: The full vocal track is used for voice cloning to capture complete voice characteristics
5. **Chord Detection**: Madmom analyzes the instrumental using high-resolution chroma features and deep learning (40-65% of processing time)
6. **Synthesis**: Coqui TTS XTTS v2 generates spoken chord names using voice cloning (15-18 seconds per unique chord)
7. **Mixing**: The synthesized vocals are overlaid onto the instrumental with -10dB reduction
8. **Download** your new track with chord vocals

## Supported Chord Types
- **Major chords**: A, B, C, D, E, F, G (and sharps/flats)
- **Minor chords**: Am, Bm, Cm, Dm, Em, Fm, Gm (and sharps/flats)
- **Seventh chords**: A7, B7, C7, D7, E7, F7, G7 (and sharps/flats)
- **Minor seventh chords**: Am7, Bm7, Cm7, Dm7, Em7, Fm7, Gm7 (and sharps/flats)
- **Major seventh chords**: Amaj7, Bmaj7, Cmaj7, Dmaj7, Emaj7, Fmaj7, Gmaj7 (and sharps/flats)
- **Diminished chords**: Adim, Bdim, Cdim, Ddim, Edim, Fdim, Gdim (and sharps/flats)
- **Augmented chords**: Aaug, Baug, Caug, Daug, Eaug, Faug, Gaug (and sharps/flats)
- **Suspended chords**: Asus2, Asus4, Bsus2, Bsus4, etc. (and sharps/flats)

## Chord Pronunciation
- **A**: "AY"
- **B**: "BEE"
- **C**: "SEE"
- **D**: "DEE"
- **E**: "EE"
- **F**: "EFF"
- **G**: "GEE"

Chord types are pronounced as: "AY MINOR", "BEE SEVENTH", "SEE MAJOR SEVENTH", etc.

## Requirements
- Python 3.11
- FFmpeg (for audio processing)
- 8GB+ RAM (for TTS models)
- GPU recommended (for faster processing)

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
5. Open `http://localhost:5001` in your browser

## Processing Time
- **3-4 minute songs**: ~5-10 minutes total processing time
- **Demucs separation**: 10-15 minutes (most time-consuming step)
- **Chord detection**: 2-3 minutes with high-sensitivity settings
- **TTS synthesis**: 15-18 seconds per unique chord
- **Real-time factor**: 12-15x (processing time vs. audio duration)

## API Endpoints
- `POST /upload` — Upload an audio file for processing
- `GET /status/<task_id>` — Check processing status with detailed progress
- `GET /download/<task_id>` — Download the processed audio
- `GET /chords/<task_id>` — Get chord progression data
- `GET /health` — Health check
- `GET /docs` — API documentation
- `POST /cancel/<task_id>` — Cancel running task
- `GET /logs/<task_id>` — Get detailed processing logs
- `GET /debug/<task_id>` — Debug task information

## Project Structure
```
chordispeak/
├── app.py           # Flask backend API
├── run.py           # Dev server launcher (port 5001)
├── index.html       # Web interface
├── requirements.txt # Python dependencies
├── version.py       # Version management script
├── VERSION          # Current version file
├── start.sh         # Startup script
├── knowledge.md     # Technical documentation
├── uploads/         # Uploaded and processed files
├── test_chord_detection.py # Chord detection debugging
└── README.md        # This file
```

## Recent Improvements (v1.0.0)

### Robust Chord Detection
- **Madmom-native audio loading**: Uses `madmom.io.audio.load_audio_file()` for better compatibility
- **Two-step approach**: Chroma feature extraction followed by chord recognition
- **Conservative filtering**: 0.5 confidence threshold, 0.5s minimum duration, 1.0s between changes
- **Multiple fallback mechanisms**: 3 different approaches for reliable chord detection
- **Comprehensive data validation**: Checks for string data, NaN/Inf values, and numeric types
- **Enhanced error handling**: Detailed logging and graceful degradation

### Web Interface Improvements
- **Static file serving**: Proper routes for Logo.png, favicon.ico, and favicon.png
- **Transparent logo**: Optimized 33KB transparent logo for better performance
- **Updated favicon**: New favicon with proper ICO and PNG formats
- **Better error reporting**: Specific error messages for each processing step

### Enhanced Progress Tracking
- **Real-time demucs progress**: Percentage tracking during vocal separation
- **Detailed step updates**: 8 distinct processing steps with percentages
- **Chord synthesis progress**: Individual chord synthesis tracking
- **Enhanced error reporting**: Specific error messages for each step

### Voice Cloning Enhancements
- **Full vocal track usage**: Complete vocal track for better voice cloning
- **PyTorch compatibility**: Fixed for PyTorch 2.6+ with monkey patch
- **TTS caching**: Results cached to avoid re-synthesis

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

## Technical Details

### Processing Pipeline
1. **Audio Preparation** (5%): Convert to WAV format
2. **Vocal Separation** (10-25%): Demucs AI separation with real-time progress
3. **Voice Sample Extraction** (30%): Extract full vocal track for cloning
4. **Chord Detection** (40-65%): Madmom deep learning analysis with robust error handling
5. **TTS Synthesis** (70-85%): Coqui TTS voice cloning per unique chord
6. **Audio Mixing** (85-100%): Overlay chord vocals onto instrumental

### Dependencies
- **Flask**: Web framework with hot-reloading
- **Madmom**: Advanced chord detection with native audio loading
- **Demucs**: High-quality vocal separation (htdemucs model)
- **Coqui TTS**: Voice synthesis with XTTS v2
- **PyTorch**: Machine learning backend
- **Librosa**: Audio processing (fallback)
- **Pydub**: Audio format conversion
- **Transformers**: 4.49.0 (compatible version)

## Known Limitations
- **Processing Time**: TTS synthesis is the slowest step
- **Memory Usage**: TTS models require significant RAM (8GB+ recommended)
- **Task Storage**: In-memory only (not persistent across restarts)
- **Audio Quality**: Depends on Demucs separation quality and vocal clarity

## License
This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
