# ChordiSpeak Knowledge Base

## Overview
ChordiSpeak is an AI-powered web application that generates spoken chord names overlaid on instrumental tracks. It uses voice cloning to make the chord announcements sound like the original singer's voice.

## Core Architecture

### Backend (Flask)
- **Framework**: Flask with hot-reloading for development
- **Port**: 5001
- **API**: RESTful endpoints for upload, status, download, and health checks
- **File Processing**: Background task processing with unique task IDs

### Frontend (HTML/JavaScript)
- **Interface**: Single-page web application
- **Status Updates**: Real-time polling for processing status
- **File Upload**: Drag-and-drop or click-to-upload
- **Progress Display**: Visual progress bar with detailed status messages
- **Chord Type Selection**: User-configurable chord detection preferences

## Processing Pipeline

### 1. Audio Preparation
- Converts uploaded audio to WAV format
- Supports MP3, WAV, FLAC, M4A input formats
- Uses pydub for audio format conversion

### 2. Vocal/Instrumental Separation
- **Tool**: Demucs AI (htdemucs model)
- **Method**: Two-stem separation (vocals vs. instrumental)
- **Output**: 
  - `vocal_track.wav` - isolated vocals
  - `instrumental_track.wav` - instrumental only
- **No Fallbacks**: Only Demucs is used, no alternative separation methods

### 3. Voice Sample Extraction
- **Method**: Uses the FULL vocal track (not just 10 seconds)
- **Purpose**: Captures complete voice characteristics for better cloning
- **Output**: `voice_sample.wav` for TTS voice cloning

### 4. Chord Detection
- **Tool**: Librosa with comprehensive chord templates
- **Method**: 
  - Chroma feature extraction with 256 hop length (high resolution)
  - Onset detection for precise timing
  - Beat tracking as fallback
  - Minimum 0.5 seconds between chord changes
  - User-configurable chord type filtering
- **Chord Templates**: 221 comprehensive templates covering all sharps and flats
- **Chord Types Supported**:
  - Major chords (always detected)
  - Minor chords
  - Seventh chords
  - Minor seventh chords
  - Major seventh chords
  - Diminished chords
  - Augmented chords
  - Suspended chords (sus2, sus4)
  - Power chords
  - Add9 chords
  - Sixth chords
  - Minor sixth chords
- **Output**: JSON with timing and chord data

### 5. Text-to-Speech Synthesis
- **Tool**: Coqui TTS XTTS v2
- **Voice Cloning**: Uses full vocal track for training
- **PyTorch Compatibility**: Fixed for PyTorch 2.6+ with monkey patch
- **Transformers Version**: Downgraded to 4.49.0 for compatibility
- **Output**: Individual WAV files for each unique chord

### 6. Audio Mixing
- **Method**: Overlays synthesized chord vocals onto instrumental
- **Volume**: Chord vocals reduced by 10dB
- **Timing**: Precise alignment with chord changes
- **Output**: Final MP3 with spoken chord overlay

## Chord-to-Speech Mapping

### Phonetic Pronunciations
- **A**: "AYE" (not "uh")
- **B**: "BEE" 
- **C**: "SEE"
- **D**: "DEE"
- **E**: "EE"
- **F**: "EFF"
- **G**: "GEE"

### Chord Types and Pronunciations
- **Major**: Just the letter (e.g., "AYE")
- **Minor**: Letter + "MINOR" (e.g., "AYE MINOR")
- **Seventh**: Letter + "SEVENTH" (e.g., "AYE SEVENTH")
- **Minor Seventh**: Letter + "MINOR SEVENTH" (e.g., "AYE MINOR SEVENTH")
- **Major Seventh**: Letter + "MAJOR SEVENTH" (e.g., "AYE MAJOR SEVENTH")
- **Diminished**: Letter + "DIMINISHED" (e.g., "BEE DIMINISHED")
- **Augmented**: Letter + "AUGMENTED" (e.g., "AYE AUGMENTED")
- **Suspended Two**: Letter + "SUSPENDED TWO" (e.g., "AYE SUSPENDED TWO")
- **Suspended Four**: Letter + "SUSPENDED FOUR" (e.g., "AYE SUSPENDED FOUR")
- **Power**: Letter + "POWER" (e.g., "AYE POWER")
- **Add Nine**: Letter + "ADD NINE" (e.g., "AYE ADD NINE")
- **Sixth**: Letter + "SIXTH" (e.g., "AYE SIXTH")
- **Minor Sixth**: Letter + "MINOR SIXTH" (e.g., "AYE MINOR SIXTH")

## User Interface Features

### Chord Type Selection
- **Interactive Checkboxes**: Users can select which chord types to detect
- **Default Selection**: Minor and seventh chords enabled by default
- **Select All/Clear All**: Convenient buttons for quick selection
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Updates**: Selections sent with file upload

### Status System

#### Backend Status Steps
1. "Preparing audio file"
2. "Splitting vocal & instrumental"
3. "Extracting voice sample"
4. "Analyzing chord pattern"
5. "Synthesizing spoken chord overlay"
6. "Overlaying spoken chords onto instrumental track"
7. "Complete"

#### Frontend Status Messages
- **Progress Bar**: Visual progress with percentage
- **Status Text**: Real-time updates from backend
- **Status Badge**: Current processing stage

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
- `POST /upload` - Upload audio file with chord type preferences
- `GET /status/<task_id>` - Check processing status
- `GET /download/<task_id>` - Download processed audio
- `GET /health` - Health check
- `GET /docs` - API documentation

### Request/Response Formats
- **Upload Request**: Multipart form with audio file and chord preferences JSON
- **Status**: JSON with status, step, progress, error
- **Download**: MP3 file stream
- **Health**: Simple OK response

## Technical Details

### Dependencies
- **Flask**: Web framework
- **Librosa**: Audio analysis and chord detection
- **Demucs**: Vocal separation
- **Coqui TTS**: Voice synthesis
- **PyTorch**: Machine learning backend
- **Pydub**: Audio processing
- **Transformers**: 4.49.0 (compatible version)

### Performance
- **Processing Time**: ~30-60 seconds for 3-4 minute songs
- **Real-time Factor**: 4-6x (TTS synthesis)
- **Memory Usage**: Moderate (TTS models loaded in memory)
- **Storage**: Temporary files in uploads directory

### Error Handling
- **Graceful Degradation**: Clear error messages
- **Status Updates**: Real-time error reporting
- **File Cleanup**: Automatic cleanup of temporary files
- **Validation**: File type and size validation

## Recent Major Improvements

### Comprehensive Chord Detection (Latest)
- **221 Chord Templates**: Complete coverage of all sharps and flats
- **13 Chord Types**: Major, minor, seventh, minor seventh, major seventh, diminished, augmented, suspended (sus2/sus4), power, add9, sixth, minor sixth
- **Enharmonic Equivalents**: C# and Db share templates, etc.
- **User Customization**: Select which chord types to detect
- **Always Major**: Major chords always detected regardless of settings

### Enhanced Chord-to-Speech Mapping
- **Improved Pronunciation**: "Seventh" instead of "seven"
- **Suspended Chords**: "Suspended two" and "suspended four"
- **Complete Coverage**: All chord types properly pronounced
- **Clear Articulation**: Easy-to-understand chord names

### User Interface Enhancements
- **Chord Type Selection**: Interactive checkboxes for chord preferences
- **Responsive Design**: Mobile-friendly interface
- **Convenience Buttons**: "Select All" and "Clear All" options
- **Real-time Integration**: Selections sent with file upload

### Backend Integration
- **Chord Preferences**: Parsed from frontend and stored with tasks
- **Filtered Detection**: Only selected chord types are detected
- **Maintained Performance**: Efficient filtering without performance impact
- **Flexible Architecture**: Easy to add new chord types

### Timing System
- **Onset Detection**: More precise chord change detection
- **Higher Resolution**: 256 hop length for better timing
- **Filtered Events**: Minimum spacing between announcements
- **Repeated Announcements**: Same chord announced multiple times if detected

### Voice Cloning
- **Full Track Usage**: Complete vocal track for training
- **Better Quality**: More natural voice cloning results
- **PyTorch Compatibility**: Fixed for latest PyTorch versions

## Known Limitations

### Technical Constraints
- **Single TTS Engine**: Only Coqui TTS supported
- **Single Separation Tool**: Only Demucs supported
- **Processing Time**: TTS synthesis is slowest step
- **Memory Usage**: TTS models require significant RAM

### Audio Quality
- **Separation Quality**: Depends on Demucs performance
- **Voice Cloning**: Quality varies with vocal clarity
- **Chord Detection**: Accuracy depends on instrumental clarity

## Future Enhancements

### Potential Improvements
- **Multiple TTS Engines**: Fallback options
- **Advanced Chord Detection**: More sophisticated algorithms
- **Real-time Processing**: Streaming audio processing
- **Batch Processing**: Multiple file upload
- **Custom Voice Training**: User-provided voice samples
- **Chord Progression Analysis**: Musical theory insights
- **Additional Chord Types**: Extended jazz chords, etc.

### Performance Optimizations
- **Caching**: TTS model caching
- **Parallel Processing**: Multi-threaded processing
- **GPU Acceleration**: CUDA support for faster processing
- **Model Optimization**: Smaller, faster models

## Troubleshooting

### Common Issues
- **PyTorch Compatibility**: Fixed with monkey patch
- **Transformers Version**: Downgraded for compatibility
- **Memory Usage**: TTS models require significant RAM
- **Processing Time**: TTS synthesis is inherently slow

### Error Recovery
- **Automatic Retry**: Failed tasks can be retried
- **Clear Error Messages**: Specific error reporting
- **Status Monitoring**: Real-time error tracking
- **File Validation**: Input file format checking

## Development Notes

### Chord Template Generation
- **Comprehensive Coverage**: All 12 notes × 13 chord types = 221 templates
- **Enharmonic Handling**: C#/Db, D#/Eb, F#/Gb, G#/Ab, A#/Bb share templates
- **Template Structure**: Chroma vectors for each chord type
- **Detection Algorithm**: Cosine similarity matching

### User Experience Design
- **Progressive Enhancement**: Works with or without chord selection
- **Intuitive Interface**: Clear labels and logical grouping
- **Responsive Layout**: Adapts to different screen sizes
- **Accessibility**: Keyboard navigation and screen reader support
