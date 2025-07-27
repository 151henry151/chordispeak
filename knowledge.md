# ChordiSpeak Knowledge Base

## Overview
ChordiSpeak is an AI-powered web application that generates spoken chord names overlaid on instrumental tracks. It uses voice cloning to make the chord announcements sound like the original singer's voice.

## Core Architecture

### Backend (Flask)
- **Framework**: Flask with hot-reloading for development
- **Port**: 5001 (configured in run.py)
- **API**: RESTful endpoints for upload, status, download, and health checks
- **File Processing**: Background task processing with unique task IDs
- **Task Storage**: In-memory dictionary (not persistent)

### Frontend (HTML/JavaScript)
- **Interface**: Single-page web application
- **Status Updates**: Real-time polling for processing status
- **File Upload**: Drag-and-drop or click-to-upload
- **Progress Display**: Visual progress bar with detailed status messages
- **Chord Type Selection**: User-configurable chord detection preferences

## Processing Pipeline

### 1. Audio Preparation
- Converts uploaded audio to WAV format using pydub
- Supports MP3, WAV, FLAC, M4A input formats
- Maximum file size: 50MB
- Output: `input.wav` in task directory

### 2. Vocal/Instrumental Separation
- **Tool**: Demucs AI (htdemucs model)
- **Method**: Two-stem separation (vocals vs. instrumental)
- **Command**: `demucs --two-stems=vocals --out {dir} --mp3 --mp3-bitrate 128 {input}`
- **Output**: 
  - `vocal_track.wav` - isolated vocals
  - `instrumental_track.wav` - instrumental only
- **Progress Tracking**: Real-time demucs progress monitoring
- **Timeout**: 15 minutes maximum processing time
- **No Fallbacks**: Only Demucs is used, no alternative separation methods

### 3. Voice Sample Extraction
- **Method**: Uses the FULL vocal track (not just a sample)
- **Purpose**: Captures complete voice characteristics for better cloning
- **Output**: `voice_sample.wav` for TTS voice cloning
- **Processing**: Direct librosa load of full vocal track

## Chord Detection

### Implementation Details
- **Tool**: Madmom (DeepChromaProcessor + DeepChromaChordRecognitionProcessor)
- **Method**: 
  - Two-step approach: Chroma feature extraction followed by chord recognition
  - High-resolution chroma features (44100Hz, 512 hop size, 50fps)
  - Deep learning-based chord recognition
  - Progress tracking with detailed step updates
  - Conservative filtering for speech synthesis
  - Confidence threshold: 0.5 (more conservative)
  - Minimum chord duration: 0.5 seconds
  - Minimum time between chords: 1.0 seconds (realistic for speech)
- **Chord Types Supported** (via madmom):
  - Major chords (maj)
  - Minor chords (min)
  - Seventh chords (7)
  - Major seventh chords (maj7)
  - Minor seventh chords (min7)
  - Diminished chords (dim)
  - Augmented chords (aug)
  - Suspended chords (sus2, sus4)
- **Output**: JSON with timing and chord data
- **Error Handling**: If Madmom fails or detects too few chords, the process fails with an error. There is no fallback to any other algorithm.

### 5. Text-to-Speech Synthesis
- **Tool**: Coqui TTS XTTS v2
- **Voice Cloning**: Uses full vocal track for training
- **PyTorch Compatibility**: Fixed for PyTorch 2.6+ with monkey patch
- **Model**: `tts_models/multilingual/multi-dataset/xtts_v2`
- **Language**: English
- **Output**: Individual WAV files for each unique chord
- **Caching**: TTS results cached to avoid re-synthesis

### 6. Audio Mixing
- **Method**: Overlays synthesized chord vocals onto instrumental
- **Volume**: Chord vocals reduced by 10dB
- **Timing**: Precise alignment with chord changes
- **Output**: Final MP3 with spoken chord overlay

## Chord-to-Speech Mapping

### Phonetic Pronunciations
- **A**: "AY" 
- **B**: "BEE" 
- **C**: "SEE"
- **D**: "DEE"
- **E**: "EE"
- **F**: "EFF"
- **G**: "GEE"

### Chord Types and Pronunciations
- **Major**: Just the letter (e.g., "AY")
- **Minor**: Letter + "MINOR" (e.g., "AY MINOR")
- **Seventh**: Letter + "SEVENTH" (e.g., "AY SEVENTH")
- **Minor Seventh**: Letter + "MINOR SEVENTH" (e.g., "AY MINOR SEVENTH")
- **Major Seventh**: Letter + "MAJOR SEVENTH" (e.g., "AY MAJOR SEVENTH")
- **Diminished**: Letter + "DIMINISHED" (e.g., "BEE DIMINISHED")
- **Augmented**: Letter + "AUGMENTED" (e.g., "AY AUGMENTED")
- **Suspended Two**: Letter + "SUSPENDED TWO" (e.g., "AY SUSPENDED TWO")
- **Suspended Four**: Letter + "SUSPENDED FOUR" (e.g., "AY SUSPENDED FOUR")
- **Power**: Letter + "POWER" (e.g., "AY POWER")
- **Add Nine**: Letter + "ADD NINE" (e.g., "AY ADD NINE")
- **Sixth**: Letter + "SIXTH" (e.g., "AY SIXTH")
- **Minor Sixth**: Letter + "MINOR SIXTH" (e.g., "AY MINOR SIXTH")

### Status System

#### Backend Status Steps
1. "Preparing audio file" (5%)
2. "Splitting vocal & instrumental" (10-25%)
3. "Extracting voice sample" (30%)
4. "Analyzing chord pattern" (40-65%)
5. "Synthesizing spoken chord overlay" (70-85%)
6. "Creating chord audio track" (85%)
7. "Overlaying spoken chords onto instrumental track" (90%)
8. "Complete" (100%)

#### Frontend Status Messages
- **Progress Bar**: Visual progress with percentage
- **Status Text**: Real-time updates from backend
- **Status Badge**: Current processing stage
- **Demucs Progress**: Special tracking for demucs percentage

## File Structure

### Project Files
```
chordispeak/
├── app.py           # Main Flask application
├── run.py           # Development server launcher
├── index.html       # Web interface
├── requirements.txt # Python dependencies
├── version.py       # Version management
├── VERSION          # Current version
├── start.sh         # Startup script
├── .gitignore       # Git ignore rules
├── knowledge.md     # Project knowledge base
└── README.md        # Project documentation
```

### Runtime Files
```
uploads/
└── {task_id}/
    ├── input.wav              # Converted input
    ├── vocal_track.wav        # Separated vocals
    ├── instrumental_track.wav  # Separated instrumental
    ├── voice_sample.wav       # Voice for cloning
    ├── chords.json            # Detected chord data
    ├── tts_{chord}.wav       # Synthesized chord audio
    └── final.mp3             # Final output
```

## API Endpoints

### Core Endpoints
- `POST /upload` - Upload audio file
- `GET /status/<task_id>` - Check processing status
- `GET /download/<task_id>` - Download processed audio
- `GET /health` - Health check
- `GET /docs` - API documentation
- `GET /debug/<task_id>` - Debug task information
- `POST /cancel/<task_id>` - Cancel running task

### Request/Response Formats
- **Upload Request**: Multipart form with audio file
- **Status**: JSON with status, step, progress, error, demucs_percentage
- **Download**: MP3 file stream
- **Health**: JSON with status, version, name

## Technical Details

### Dependencies
- **Flask**: Web framework
- **Librosa**: Audio loading and processing (not used for chord detection)
- **Madmom**: Advanced chord detection (only algorithm used)
- **Demucs**: Vocal separation
- **Coqui TTS**: Voice synthesis
- **PyTorch**: Machine learning backend
- **Pydub**: Audio processing
- **Transformers**: 4.49.0 (compatible version)
- **Scipy**: Audio processing
- **Numpy**: Numerical computing

### Performance
- **Processing Time**: ~5-10 minutes for 3-4 minute songs
- **Real-time Factor**: 12-15x (TTS synthesis)
- **Memory Usage**: High (TTS models loaded in memory)
- **Storage**: Temporary files in uploads directory
- **Demucs Time**: ~10-15 minutes for vocal separation

### Error Handling
- **Graceful Degradation**: Clear error messages
- **Status Updates**: Real-time error reporting
- **File Cleanup**: Automatic cleanup of temporary files
- **Validation**: File type and size validation
- **Timeout Protection**: 15-minute timeout for demucs
- **Chord Detection**: If Madmom fails, the process fails (no fallback)

## Recent Major Improvements

### Advanced Chord Detection (Latest)
- **Only Madmom**: Deep learning chord detection with no fallback
- **Progress Tracking**: Detailed progress updates during chord detection
- **Confidence Filtering**: Only high-confidence chords (0.6+)
- **Duration Filtering**: Minimum 1.0 second chord duration
- **Smoothing**: Median filtering to reduce rapid switching
- **Timing Optimization**: Minimum 1.0 seconds between chord changes

### Enhanced Progress Tracking
- **Demucs Progress**: Real-time demucs percentage tracking
- **Detailed Steps**: 8 distinct processing steps with percentages
- **Chord Synthesis Progress**: Individual chord synthesis tracking
- **Error Reporting**: Specific error messages for each step

### Voice Cloning Improvements
- **Full Track Usage**: Complete vocal track for training
- **Better Quality**: More natural voice cloning results
- **PyTorch Compatibility**: Fixed for latest PyTorch versions
- **Caching**: TTS results cached to avoid re-synthesis

### Audio Processing Enhancements
- **High-Quality Separation**: Demucs htdemucs model
- **Precise Timing**: Onset detection for chord changes
- **Volume Control**: -10dB reduction for chord vocals
- **Format Support**: MP3, WAV, FLAC, M4A input formats

## Known Limitations

### Technical Constraints
- **Single TTS Engine**: Only Coqui TTS supported
- **Single Separation Tool**: Only Demucs supported
- **Processing Time**: TTS synthesis is slowest step (~15-18 seconds per chord)
- **Memory Usage**: TTS models require significant RAM
- **Task Storage**: In-memory only (not persistent across restarts)
- **Chord Detection**: If Madmom fails, the process fails (no fallback)

### Audio Quality
- **Separation Quality**: Depends on Demucs performance
- **Voice Cloning**: Quality varies with vocal clarity
- **Chord Detection**: Accuracy depends on instrumental clarity
- **Processing Time**: Long processing times for complex songs

### Performance Constraints
- **Demucs Time**: 10-15 minutes for vocal separation
- **TTS Synthesis**: 15-18 seconds per unique chord
- **Memory Usage**: High memory requirements for TTS models
- **CPU Intensive**: Heavy computational requirements

## Future Enhancements

### Potential Improvements
- **Multiple TTS Engines**: Fallback options
- **Advanced Chord Detection**: More sophisticated algorithms
- **Real-time Processing**: Streaming audio processing
- **Batch Processing**: Multiple file upload
- **Custom Voice Training**: User-provided voice samples
- **Chord Progression Analysis**: Musical theory insights
- **Additional Chord Types**: Extended jazz chords, etc.
- **Persistent Storage**: Database for task persistence
- **GPU Acceleration**: CUDA support for faster processing

### Performance Optimizations
- **Caching**: TTS model caching
- **Parallel Processing**: Multi-threaded processing
- **Model Optimization**: Smaller, faster models
- **Incremental Processing**: Process in chunks

## Troubleshooting

### Common Issues
- **PyTorch Compatibility**: Fixed with monkey patch
- **Transformers Version**: Downgraded for compatibility
- **Memory Usage**: TTS models require significant RAM
- **Processing Time**: TTS synthesis is inherently slow
- **Demucs Timeout**: 15-minute timeout for vocal separation
- **Chord Detection**: If Madmom fails, the process fails (no fallback)

### Error Recovery
- **Automatic Retry**: Failed tasks can be retried
- **Clear Error Messages**: Specific error reporting
- **Status Monitoring**: Real-time error tracking
- **File Validation**: Input file format checking
- **Task Cancellation**: Users can cancel running tasks

## Development Notes

### Chord Detection Architecture
- **Only Madmom**: Deep learning approach with no fallback
- **Progress Tracking**: Real-time updates during processing
- **Error Handling**: Graceful error reporting if Madmom fails

### User Experience Design
- **Progressive Enhancement**: Works with or without chord selection
- **Intuitive Interface**: Clear labels and logical grouping
- **Responsive Layout**: Adapts to different screen sizes
- **Accessibility**: Keyboard navigation and screen reader support
- **Real-time Feedback**: Detailed progress updates

### Performance Considerations
- **Memory Management**: TTS models loaded per task
- **Processing Pipeline**: Sequential processing with progress tracking
- **File Management**: Temporary files in task directories
- **Error Handling**: Comprehensive error reporting and recovery

Last stable commit: https://github.com/151henry151/chordispeak/commit/d6e32fae9cadc996fd9b9e95df5fcb021e977ae6