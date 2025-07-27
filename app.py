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
"""

import os
import uuid
import json
import re
from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
from werkzeug.utils import secure_filename
import subprocess
import tempfile
import shutil
from threading import Thread
import time

# GPU Optimization: Set environment variables for optimal GPU performance
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')  # Non-blocking CUDA operations
os.environ.setdefault('TORCH_CUDNN_V8_API_ENABLED', '1')  # Enable cuDNN v8

# Lazy imports for heavy dependencies
librosa = None
np = None
AudioSegment = None
write_wav = None
torch = None
TTS = None
TTS_AVAILABLE = False

def get_version():
    """Read version from VERSION file"""
    try:
        with open('VERSION', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return "1.0.0"  # fallback version

VERSION = get_version()

def lazy_import_audio_deps():
    """Lazy import audio processing dependencies"""
    global librosa, np, AudioSegment, write_wav
    
    if librosa is None:
        try:
            import librosa
            import numpy as np
            from pydub import AudioSegment
            from scipy.io.wavfile import write as write_wav
            
            # Fix for madmom Python 3.11 compatibility issue
            import collections
            if not hasattr(collections, 'MutableSequence'):
                from collections.abc import MutableSequence
                collections.MutableSequence = MutableSequence

            # Fix for madmom numpy compatibility issue
            if not hasattr(np, 'float'):
                np.float = float
            if not hasattr(np, 'int'):
                np.int = int
            if not hasattr(np, 'complex'):
                np.complex = complex
                
        except ImportError as e:
            print(f"Warning: Audio processing dependencies not available: {e}")
            return False
    return True

def lazy_import_tts():
    """Lazy import TTS dependencies"""
    global torch, TTS, TTS_AVAILABLE
    
    if TTS is None:
        try:
            import torch
            from TTS.api import TTS
            TTS_AVAILABLE = True
            
            # GPU detection and logging
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device count: {torch.cuda.device_count()}")
                print(f"Current CUDA device: {torch.cuda.current_device()}")
                print(f"CUDA device name: {torch.cuda.get_device_name()}")
                print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                print(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            else:
                print("CUDA not available - using CPU")
                
        except ImportError:
            TTS_AVAILABLE = False
            print("Coqui TTS not available. Please install Coqui TTS (XTTS v2) to use this app.")
    
    return TTS_AVAILABLE

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

# Pronunciation strategy configuration
# Options: 'ipa', 'dots', 'words', 'nato', 'simple', 'spelled'
PRONUNCIATION_STRATEGY = 'dots'  # Try dots strategy for clearer letter pronunciation

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Task storage (in production, use Redis or database)
tasks = {}

# Debug mode configuration
DEBUG_MODE = True  # Set to True to enable debug logging
task_logs = {}  # Store logs for each task

def log_debug(task_id, message):
    """Log debug message for a specific task"""
    if DEBUG_MODE and task_id:
        if task_id not in task_logs:
            task_logs[task_id] = []
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        task_logs[task_id].append(log_entry)
        # Keep only last 100 log entries to prevent memory issues
        if len(task_logs[task_id]) > 100:
            task_logs[task_id] = task_logs[task_id][-100:]
        print(f"[DEBUG {task_id}] {message}")
    else:
        # Always print debug messages even if DEBUG_MODE is False or task_id is None
        print(f"[DEBUG {task_id}] {message}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def chord_to_ipa_phonemes(chord):
    """Convert chord notation to phonetic spellings for precise pronunciation"""
    # Strategy 1: IPA Phonemes (most precise)
    letter_phonemes_ipa = {
        'A': 'ˈeɪ',  # IPA for "A"
        'B': 'ˈbiː',  # IPA for "B"
        'C': 'ˈsiː',  # IPA for "C"
        'D': 'ˈdiː',  # IPA for "D"
        'E': 'ˈiː',   # IPA for "E"
        'F': 'ˈɛf',   # IPA for "F"
        'G': 'ˈdʒiː'  # IPA for "G"
    }
    
    # Strategy 2: Spelling with dots (clear separation)
    letter_phonemes_dots = {
        'A': 'A.',
        'B': 'B.',
        'C': 'C.',
        'D': 'D.',
        'E': 'E.',
        'F': 'F.',
        'G': 'G.'
    }
    
    # Strategy 3: Contextual words (natural pronunciation)
    letter_phonemes_words = {
        'A': 'A as in apple',
        'B': 'B as in boy',
        'C': 'C as in cat',
        'D': 'D as in dog',
        'E': 'E as in easy',
        'F': 'F as in fun',
        'G': 'G as in go'
    }
    
    # Strategy 4: NATO phonetic alphabet
    letter_phonemes_nato = {
        'A': 'Alpha',
        'B': 'Bravo',
        'C': 'Charlie',
        'D': 'Delta',
        'E': 'Echo',
        'F': 'Foxtrot',
        'G': 'Golf'
    }
    
    # Strategy 5: Simple spellings (current approach)
    letter_phonemes_simple = {
        'A': 'AY',
        'B': 'BEE', 
        'C': 'SEE',
        'D': 'DEE',
        'E': 'EE',
        'F': 'EFF',
        'G': 'GEE'
    }
    
    # Choose strategy based on configuration
    if PRONUNCIATION_STRATEGY == 'ipa':
        letter_phonemes = letter_phonemes_ipa
    elif PRONUNCIATION_STRATEGY == 'dots':
        letter_phonemes = letter_phonemes_dots
    elif PRONUNCIATION_STRATEGY == 'words':
        letter_phonemes = letter_phonemes_words
    elif PRONUNCIATION_STRATEGY == 'nato':
        letter_phonemes = letter_phonemes_nato
    elif PRONUNCIATION_STRATEGY == 'spelled':
        letter_phonemes = {
            'A': 'A Y', 'B': 'B E E', 'C': 'C E E', 'D': 'D E E', 
            'E': 'E E', 'F': 'E F F', 'G': 'G E E'
        }
    else:  # Default to simple
        letter_phonemes = letter_phonemes_simple
    
    # Modifier phonemes
    modifiers = {
        '#': 'SHARP',  # sharp
        'b': 'FLAT',  # flat
        'm': 'MINOR',  # minor
        '7': 'SEVENTH',  # seventh
        'maj7': 'MAJOR SEVENTH',  # major seventh
        'm7': 'MINOR SEVENTH',  # minor seventh
        'dim': 'DIMINISHED',  # diminished
        'aug': 'AUGMENTED',  # augmented
        'sus': 'SUSPENDED',  # suspended
        'sus2': 'SUSPENDED TWO',  # suspended two
        'sus4': 'SUSPENDED FOUR',  # suspended four
        '6': 'SIXTH',  # sixth
        'm6': 'MINOR SIXTH',  # minor sixth
        'add9': 'ADD NINE',  # add nine
        '5': 'POWER'  # power
    }
    
    # Handle single letter chords
    if len(chord) == 1 and chord in letter_phonemes:
        return letter_phonemes[chord]
    
    # Handle chords with sharps/flats
    if '#' in chord:
        base = chord.split('#')[0]
        if base in letter_phonemes:
            return f"{letter_phonemes[base]} {modifiers['#']}"
    elif 'b' in chord and len(chord) > 1:
        base = chord.split('b')[0]
        if base in letter_phonemes:
            return f"{letter_phonemes[base]} {modifiers['b']}"
    
    # Handle special chord types
    for suffix, phoneme in modifiers.items():
        if chord.endswith(suffix):
            base = chord[:-len(suffix)]
            if base in letter_phonemes:
                return f"{letter_phonemes[base]} {phoneme}"
    
    # Default: return the letter phoneme if available
    if chord in letter_phonemes:
        return letter_phonemes[chord]
    
    # Fallback to regular text
    return chord

def format_phonemes_for_tts(phoneme_text):
    """Format phonetic text for TTS input using simple spellings"""
    # For simple phonetic spellings, just return as-is
    # The TTS should handle these better than IPA
    return phoneme_text

def format_chord_for_tts(chord_text):
    """Format chord text for better TTS pronunciation using phonetic spellings"""
    # For simple phonetic spellings, just return as-is
    # The TTS should handle these better than IPA
    return chord_text

def detect_vocal_content(audio_path, min_vocal_duration=20.0):
    """Detect if audio contains sufficient vocal content (at least min_vocal_duration seconds)"""
    print(f"detect_vocal_content called with audio_path: {audio_path}")
    
    if not lazy_import_audio_deps():
        print("Audio processing dependencies not available")
        return False
        
    try:
        print(f"Loading audio with librosa from: {audio_path}")
        y, sr = librosa.load(audio_path)
        print(f"Audio loaded successfully: shape={y.shape}, sr={sr}")
        
        # Calculate total duration
        duration = len(y) / sr
        print(f"Total audio duration: {duration:.2f} seconds")
        
        # Simple vocal detection: look for segments with significant energy
        # This is a basic approach - in practice you might want more sophisticated vocal detection
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # Calculate RMS energy for each frame
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Normalize RMS
        rms_normalized = rms / np.max(rms) if np.max(rms) > 0 else rms
        
        # Threshold for vocal activity (adjust as needed)
        vocal_threshold = 0.1
        
        # Find frames with vocal activity
        vocal_frames = rms_normalized > vocal_threshold
        
        # Calculate vocal duration
        vocal_duration = np.sum(vocal_frames) * hop_length / sr
        print(f"Detected vocal duration: {vocal_duration:.2f} seconds")
        
        has_sufficient_vocals = vocal_duration >= min_vocal_duration
        print(f"Has sufficient vocals ({min_vocal_duration}s): {has_sufficient_vocals}")
        
        return has_sufficient_vocals
        
    except Exception as e:
        print(f"Vocal content detection error: {e}")
        import traceback
        traceback.print_exc()
        return False

def extract_voice_sample(vocals_path, sample_duration=None):
    """Extract voice sample from separated vocals for voice cloning"""
    print(f"extract_voice_sample called with vocals_path: {vocals_path}")
    
    if not lazy_import_audio_deps():
        print("Audio processing dependencies not available")
        return None, None
        
    try:
        print(f"Loading vocals with librosa from: {vocals_path}")
        # Use the full vocal track for better voice cloning
        y, sr = librosa.load(vocals_path)
        print(f"Vocals loaded successfully: shape={y.shape}, sr={sr}")
        
        # Return the entire vocal track
        return y, sr
        
    except Exception as e:
        print(f"Voice sample extraction error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def synthesize_chord_speech_coqui(text, voice_sample_path, output_path):
    """Generate speech using Coqui XTTS v2 voice cloning with phoneme support"""
    if not lazy_import_tts():
        raise RuntimeError("Coqui TTS not available. Please install Coqui TTS (XTTS v2).")
    try:
        # Fix for PyTorch 2.6+ weights_only issue - monkey patch torch.load
        original_torch_load = torch.load
        
        def patched_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        torch.load = patched_torch_load
        
        # Initialize TTS with XTTS v2 model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"TTS using device: {device}")
        
        # GPU memory optimization
        if device == "cuda":
            # Clear GPU cache before loading model
            torch.cuda.empty_cache()
            print(f"GPU memory before TTS model load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Pre-accept Coqui TTS license to avoid interactive prompts
        os.environ['COQUI_TOS_AGREED'] = '1'
        
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        
        if device == "cuda":
            print(f"GPU memory after TTS model load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Convert chord to phonemes
        phoneme_text = chord_to_ipa_phonemes(text)
        formatted_phonemes = format_phonemes_for_tts(phoneme_text)
        
        print(f"Using phoneme approach: '{text}' -> '{formatted_phonemes}'")
        
        # Generate speech with phonemes
        tts.tts_to_file(
            text=formatted_phonemes,
            speaker_wav=voice_sample_path,
            language="en",
            file_path=output_path
        )
        
        # GPU memory cleanup after synthesis
        if device == "cuda":
            torch.cuda.empty_cache()
            print(f"GPU memory after TTS synthesis: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Restore original torch.load
        torch.load = original_torch_load
        return True
    except Exception as e:
        print(f"Coqui TTS synthesis error: {e}")
        return False

def synthesize_chord_speech(text, voice_sample_path, output_path):
    """Generate speech using only Coqui XTTS v2 voice cloning"""
    if not lazy_import_tts():
        raise RuntimeError("Coqui TTS not available. Please install Coqui TTS (XTTS v2).")
    if not voice_sample_path or not os.path.exists(voice_sample_path):
        raise RuntimeError("Voice sample for cloning not found. Cannot synthesize without a reference voice.")
    return synthesize_chord_speech_coqui(text, voice_sample_path, output_path)

def detect_chords(audio_file, chord_types=None, task_id=None):
    """Detect chords from audio file using madmom for accurate chord recognition"""
    print(f"[TASK {task_id}] Starting chord detection")
    
    if not lazy_import_audio_deps():
        raise RuntimeError("Audio processing dependencies not available. Cannot perform chord detection.")
    
    # Validate audio file
    print(f"[TASK {task_id}] Validating audio file: {audio_file}")
    try:
        import os
        if not os.path.exists(audio_file):
            raise RuntimeError(f"Audio file does not exist: {audio_file}")
        
        file_size = os.path.getsize(audio_file)
        if file_size == 0:
            raise RuntimeError(f"Audio file is empty: {audio_file}")
        
        print(f"[TASK {task_id}] Audio file validation passed: size={file_size} bytes")
        
    except Exception as validation_error:
        print(f"[TASK {task_id}] Audio file validation failed: {validation_error}")
        raise RuntimeError(f"Audio file validation failed: {validation_error}")
    
    try:
        # Import madmom compatibility layer
        from madmom_compat import detect_chords_simple
        
        # Update progress: Starting chord detection (40-50%)
        if task_id and task_id in tasks:
            tasks[task_id]['step'] = 'Analyzing chord pattern'
            tasks[task_id]['progress'] = 40
            print(f"[TASK {task_id}] Progress: 40% - Starting chord detection")
        
        # Use madmom's standard chord detection
        print(f"[TASK {task_id}] Using madmom chord detection...")
        chords = detect_chords_simple(audio_file, task_id)
        
        if not chords or len(chords) == 0:
            raise RuntimeError("No chords detected. This could indicate an issue with the audio file or chord detection.")
        
        print(f"[TASK {task_id}] Raw madmom output: {len(chords)} chord segments")
        
        # Process madmom output format
        # Madmom returns list of [start_time, end_time, chord_label, confidence]
        valid_chords = []
        filtered_out_count = 0
        min_confidence = 0.5
        min_chord_duration = 0.5
        min_time_between_chords = 1.0
        
        print(f"[TASK {task_id}] Processing {len(chords)} chord segments from madmom...")
        
        for i, chord_data in enumerate(chords):
            try:
                # Madmom format: [start_time, end_time, chord_label, confidence]
                if len(chord_data) >= 4:
                    start_time = float(chord_data[0])
                    end_time = float(chord_data[1])
                    chord_label = str(chord_data[2])
                    confidence = float(chord_data[3])
                elif len(chord_data) == 3:
                    start_time = float(chord_data[0])
                    end_time = float(chord_data[1])
                    chord_label = str(chord_data[2])
                    confidence = 1.0  # Default confidence
                else:
                    print(f"[TASK {task_id}] Skipping invalid chord data format: {chord_data}")
                    filtered_out_count += 1
                    continue
                
                # Skip 'N' (no chord) detections
                if chord_label == 'N':
                    filtered_out_count += 1
                    continue
                
                # Filter by confidence threshold
                if confidence < min_confidence:
                    print(f"[TASK {task_id}] Filtered out low confidence chord: {chord_label} at {start_time:.2f}s (confidence: {confidence:.3f} < {min_confidence})")
                    filtered_out_count += 1
                    continue
                
                # Calculate chord duration
                chord_duration = end_time - start_time
                
                # Skip very short chord detections
                if chord_duration < min_chord_duration:
                    print(f"[TASK {task_id}] Filtered out short chord: {chord_label} at {start_time:.2f}s (duration: {chord_duration:.3f}s < {min_chord_duration}s)")
                    filtered_out_count += 1
                    continue
                
                # Convert madmom chord labels to our format
                chord_name = convert_madmom_chord(chord_label)
                
                print(f"[TASK {task_id}] Processing chord: {chord_label} -> {chord_name} at {start_time:.2f}-{end_time:.2f}s (confidence: {confidence:.3f})")
                
                valid_chords.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'chord': chord_name,
                    'confidence': confidence,
                    'duration': chord_duration
                })
                
            except Exception as e:
                print(f"[TASK {task_id}] Error processing chord data {chord_data}: {e}")
                filtered_out_count += 1
                continue
        
        print(f"[TASK {task_id}] Successfully processed {len(valid_chords)} chord segments (filtered out {filtered_out_count})")
        
        if not valid_chords:
            raise RuntimeError("No valid chords detected. This could indicate an issue with the audio file or chord detection.")
        
        # Sort by start time
        valid_chords.sort(key=lambda x: x['start_time'])
        
        # Apply smoothing and timing optimization for speech synthesis
        print(f"[TASK {task_id}] Applying chord smoothing and timing optimization...")
        
        # Apply median filtering to reduce rapid switching
        window_size = 2
        smoothed_chords = []
        
        for i in range(len(valid_chords)):
            # Get window of chords around current position
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(valid_chords), i + window_size // 2 + 1)
            window = valid_chords[start_idx:end_idx]
            
            # Find most common chord in window
            chord_counts = {}
            for chord_info in window:
                chord = chord_info['chord']
                chord_counts[chord] = chord_counts.get(chord, 0) + 1
            
            # Use the most common chord in the window
            most_common_chord = max(chord_counts.items(), key=lambda x: x[1])[0]
            
            # Only add if it's different from the last added chord
            if not smoothed_chords or most_common_chord != smoothed_chords[-1]['chord']:
                smoothed_chords.append({
                    'time': valid_chords[i]['start_time'],
                    'chord': most_common_chord,
                    'speech': chord_to_ipa_phonemes(most_common_chord),
                    'confidence': valid_chords[i]['confidence'],
                    'duration': valid_chords[i]['duration']
                })
        
        # Final pass: ensure minimum time between chord changes
        final_chords = []
        for chord_info in smoothed_chords:
            if not final_chords or (chord_info['time'] - final_chords[-1]['time']) >= min_time_between_chords:
                final_chords.append(chord_info)
            else:
                print(f"[TASK {task_id}] Filtered out rapid chord change: {chord_info['chord']} at {chord_info['time']:.2f}s (too close to previous)")
        
        print(f"[TASK {task_id}] Final result: {len(final_chords)} chord changes")
        for chord_info in final_chords:
            print(f"[TASK {task_id}]   {chord_info['chord']} at {chord_info['time']:.2f}s")
        
        # Update progress: Chord detection complete (65%)
        if task_id and task_id in tasks:
            tasks[task_id]['step'] = 'Analyzing chord pattern (complete)'
            tasks[task_id]['progress'] = 65
            print(f"[TASK {task_id}] Progress: 65% - Chord detection and processing complete")
        
        return final_chords
        
    except Exception as e:
        print(f"[TASK {task_id}] ERROR in chord detection: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Chord detection failed: {e}")

def convert_madmom_chord(madmom_label):
    """Convert madmom chord labels to our chord format"""
    # madmom uses labels like 'C:maj', 'C:min', 'C:7', etc.
    if ':' not in madmom_label:
        return madmom_label
    
    root, quality = madmom_label.split(':')
    
    # Convert quality to our format
    if quality == 'maj':
        return root
    elif quality == 'min':
        return root + 'm'
    elif quality == '7':
        return root + '7'
    elif quality == 'maj7':
        return root + 'maj7'
    elif quality == 'min7':
        return root + 'm7'
    elif quality == 'dim':
        return root + 'dim'
    elif quality == 'aug':
        return root + 'aug'
    elif quality == 'sus2':
        return root + 'sus2'
    elif quality == 'sus4':
        return root + 'sus4'
    else:
        return madmom_label  # Return as-is for unknown qualities

def synthesize_chord_speech(text, voice_sample_path, output_path):
    """Generate speech using only Coqui XTTS v2 voice cloning"""
    if not lazy_import_tts():
        raise RuntimeError("Coqui TTS not available. Please install Coqui TTS (XTTS v2).")
    if not voice_sample_path or not os.path.exists(voice_sample_path):
        raise RuntimeError("Voice sample for cloning not found. Cannot synthesize without a reference voice.")
    return synthesize_chord_speech_coqui(text, voice_sample_path, output_path)

def detect_chords(audio_file, chord_types=None, task_id=None):
    """Detect chords from audio file using madmom for accurate chord recognition with progress tracking"""
    print(f"[TASK {task_id}] Starting detect_chords function")
    
    if not lazy_import_audio_deps():
        raise RuntimeError("Audio processing dependencies not available. Cannot perform chord detection.")
    
    # Comprehensive audio file validation
    print(f"[TASK {task_id}] Validating audio file: {audio_file}")
    try:
        import os
        if not os.path.exists(audio_file):
            raise RuntimeError(f"Audio file does not exist: {audio_file}")
        
        file_size = os.path.getsize(audio_file)
        if file_size == 0:
            raise RuntimeError(f"Audio file is empty: {audio_file}")
        
        print(f"[TASK {task_id}] Audio file validation passed: size={file_size} bytes")
        
        # Try to validate with soundfile first
        import soundfile as sf
        try:
            info = sf.info(audio_file)
            print(f"[TASK {task_id}] Soundfile validation: samplerate={info.samplerate}, channels={info.channels}, duration={info.duration:.2f}s")
            if info.duration <= 0:
                raise RuntimeError(f"Audio file has zero duration: {audio_file}")
        except Exception as sf_error:
            print(f"[TASK {task_id}] Warning: Soundfile validation failed: {sf_error}")
            # Continue with librosa validation
        
    except Exception as validation_error:
        print(f"[TASK {task_id}] Audio file validation failed: {validation_error}")
        raise RuntimeError(f"Audio file validation failed: {validation_error}")
        
    try:
        import time
        print(f"[TASK {task_id}] Importing madmom modules...")
        from madmom.features.chords import DeepChromaChordRecognitionProcessor
        print(f"[TASK {task_id}] Madmom modules imported successfully")
        
        # Initialize madmom chord detector
        print(f"[TASK {task_id}] Initializing madmom chord detector...")
        
        # Debug: Print audio file info
        import soundfile as sf
        try:
            info = sf.info(audio_file)
            print(f"[TASK {task_id}] Audio file info: samplerate={info.samplerate}, channels={info.channels}, duration={info.duration:.2f}s, format={info.format}, subtype={info.subtype}")
        except Exception as e:
            print(f"[TASK {task_id}] Could not get audio file info with soundfile: {e}")
        
        # Get audio duration for progress estimation
        # Use madmom's own audio loading for better compatibility
        print(f"[TASK {task_id}] Loading audio with madmom's audio loader...")
        try:
            from madmom.io.audio import load_audio_file
            y, sr = load_audio_file(audio_file, sample_rate=44100, num_channels=1, dtype=np.float32)
            audio_duration = len(y) / sr
            print(f"[TASK {task_id}] Madmom audio loader: duration={audio_duration:.2f}s, sr={sr}, shape={y.shape}, dtype={y.dtype}")
        except Exception as madmom_load_error:
            print(f"[TASK {task_id}] Madmom audio loading failed: {madmom_load_error}")
            print(f"[TASK {task_id}] Falling back to librosa...")
            # Fallback to librosa
            y, sr = librosa.load(audio_file, sr=None, mono=True, dtype=np.float32)
            audio_duration = len(y) / sr
            print(f"[TASK {task_id}] Librosa fallback: duration={audio_duration:.2f}s, sr={sr}, shape={y.shape}, dtype={y.dtype}")
        
        # CRITICAL: Check for string data in audio (this causes the NumPy error)
        if y.dtype.kind in ['U', 'S']:
            print(f"[TASK {task_id}] ERROR: Audio data contains strings! dtype={y.dtype}")
            print(f"[TASK {task_id}] This will cause the NumPy multiply error. Attempting to fix...")
            
            # Try to reload with different parameters
            try:
                print(f"[TASK {task_id}] Attempting to reload audio with explicit numeric conversion...")
                y, sr = librosa.load(audio_file, sr=44100, mono=True, dtype=np.float32)
                
                # Check again
                if y.dtype.kind in ['U', 'S']:
                    print(f"[TASK {task_id}] ERROR: Still getting string data after reload!")
                    raise RuntimeError("Audio file contains string data instead of numeric data")
                else:
                    print(f"[TASK {task_id}] Successfully reloaded audio with numeric data: dtype={y.dtype}")
            except Exception as reload_error:
                print(f"[TASK {task_id}] Failed to reload audio: {reload_error}")
                raise RuntimeError(f"Audio file appears to be corrupted or contains invalid data: {reload_error}")
        
        # Additional validation: Check for NaN or Inf values
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            print(f"[TASK {task_id}] WARNING: Audio contains NaN or Inf values, cleaning...")
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Ensure audio data is float32 for madmom compatibility
        if y.dtype != np.float32:
            print(f"[TASK {task_id}] Converting audio dtype from {y.dtype} to float32")
            y = y.astype(np.float32)
        
        # Additional NumPy compatibility fixes for madmom
        print(f"[TASK {task_id}] Applying NumPy compatibility fixes...")
        if not hasattr(np, 'float'):
            np.float = float
        if not hasattr(np, 'int'):
            np.int = int
        if not hasattr(np, 'complex'):
            np.complex = complex
        
        # Ensure audio is properly formatted for madmom
        print(f"[TASK {task_id}] Preparing audio for madmom processing...")
        # Normalize audio to prevent overflow issues
        if np.max(np.abs(y)) > 1.0:
            y = y / np.max(np.abs(y))
        
        # Ensure audio is contiguous in memory
        y = np.ascontiguousarray(y, dtype=np.float32)
        
        # Final validation before processing
        print(f"[TASK {task_id}] Final audio validation:")
        print(f"[TASK {task_id}]   - Shape: {y.shape}")
        print(f"[TASK {task_id}]   - Dtype: {y.dtype}")
        print(f"[TASK {task_id}]   - Range: [{np.min(y):.6f}, {np.max(y):.6f}]")
        print(f"[TASK {task_id}]   - Has NaN: {np.any(np.isnan(y))}")
        print(f"[TASK {task_id}]   - Has Inf: {np.any(np.isinf(y))}")
        print(f"[TASK {task_id}]   - Is numeric: {np.issubdtype(y.dtype, np.number)}")
        
        if not np.issubdtype(y.dtype, np.number):
            raise RuntimeError("Audio data is not numeric - cannot proceed with chord detection")
        
        # Initialize madmom chord detection using the recommended two-step approach
        print(f"[TASK {task_id}] Initializing madmom processors...")
        try:
            # Add debug info about madmom version and dependencies
            import madmom
            print(f"[TASK {task_id}] Madmom version: {madmom.__version__}")
            
            # Check NumPy version compatibility
            print(f"[TASK {task_id}] NumPy version: {np.__version__}")
            print(f"[TASK {task_id}] Librosa version: {librosa.__version__}")
            
            # Check for known compatibility issues
            if hasattr(madmom, '__version__') and madmom.__version__:
                print(f"[TASK {task_id}] Madmom version check completed")
            else:
                print(f"[TASK {task_id}] Warning: Could not determine madmom version")
            
            # Check what processors are actually available
            try:
                from madmom.features.chords import DeepChromaProcessor
                print(f"[TASK {task_id}] DeepChromaProcessor is available")
                chroma_processor_available = True
            except ImportError:
                print(f"[TASK {task_id}] DeepChromaProcessor not available, using single-step approach")
                chroma_processor_available = False
            
            # Initialize madmom processors with recommended parameters
            # Step 1: Chroma feature extraction
            if chroma_processor_available:
                chroma_processor = DeepChromaProcessor(
                    sample_rate=44100,  # Standard sample rate
                    hop_size=512,       # Frame hop size for good temporal resolution
                    fps=50,            # 50 frames per second for smooth detection
                    num_classes=25      # Standard number of chord classes
                )
                print(f"[TASK {task_id}] Chroma processor initialized successfully")
            else:
                print(f"[TASK {task_id}] Using single-step chord detection approach")
            
            # Initialize chord detector
            chord_detector = DeepChromaChordRecognitionProcessor()
            print(f"[TASK {task_id}] Chord detector initialized successfully")
            
        except Exception as e:
            print(f"[TASK {task_id}] ERROR initializing madmom processor: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize madmom processor: {e}")
        
        # Start timing for progress estimation
        start_time = time.time()
        
        # Initialize variables for chord processing with more conservative thresholds
        filtered_out_count = 0
        min_confidence = 0.5  # More conservative confidence threshold
        min_chord_duration = 0.5  # Longer minimum chord duration
        min_time_between_chords = 1.0  # More realistic for speech synthesis
        valid_chords = []
        
        # Update progress: Starting chord detection (40-50%)
        if task_id and task_id in tasks:
            tasks[task_id]['step'] = 'Analyzing chord pattern'
            tasks[task_id]['progress'] = 40
            print(f"[TASK {task_id}] Progress: 40% - Starting chord detection")
        
        # Process the audio file with the chord detector
        print(f"[TASK {task_id}] Starting chord detection...")
        try:
            print(f"[TASK {task_id}] Calling chord detection with audio_file: {audio_file}")
            # Add debug info about the audio file
            import os
            if os.path.exists(audio_file):
                file_size = os.path.getsize(audio_file)
                print(f"[TASK {task_id}] Audio file size: {file_size} bytes")
                
                # Check if file is not empty
                if file_size == 0:
                    print(f"[TASK {task_id}] ERROR: Audio file is empty!")
                    raise RuntimeError("Audio file is empty")
                
                # Try to validate the audio file format
                try:
                    import soundfile as sf
                    info = sf.info(audio_file)
                    print(f"[TASK {task_id}] Audio file format: {info.format}, subtype: {info.subtype}")
                    if info.duration <= 0:
                        print(f"[TASK {task_id}] ERROR: Audio file has zero duration!")
                        raise RuntimeError("Audio file has zero duration")
                except Exception as sf_error:
                    print(f"[TASK {task_id}] Warning: Could not validate audio file with soundfile: {sf_error}")
            else:
                print(f"[TASK {task_id}] ERROR: Audio file does not exist: {audio_file}")
                raise RuntimeError(f"Audio file not found: {audio_file}")
            
            try:
                # Debug: Check what madmom is receiving
                print(f"[TASK {task_id}] Audio file path: {audio_file}")
                print(f"[TASK {task_id}] Audio file type: {type(audio_file)}")
                
                # Use the main chord detection approach
                print(f"[TASK {task_id}] Using main chord detection approach...")
                chords = chord_detector(audio_file)
                print(f"[TASK {task_id}] Chord detection completed successfully")
                
            except Exception as chord_error:
                print(f"[TASK {task_id}] ERROR during chord detection call: {chord_error}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Chord detection failed: {chord_error}")
                
        except Exception as e:
            print(f"[TASK {task_id}] ERROR in chord detection: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Chord detection failed: {e}")
        
        # Save chord data
        chords_file = os.path.join(task_dir, 'chords.json')
        with open(chords_file, 'w') as f:
            json.dump(chords, f)
        print(f"Chord data saved: {chords_file}")
        
        # Step 5: Voice synthesis using voice cloning
        print(f"\n=== [TASK {task_id}] STEP 5: VOICE SYNTHESIS ===")
        tasks[task_id]['step'] = 'Synthesizing spoken chord overlay'
        tasks[task_id]['progress'] = 70
        print(f"[TASK {task_id}] Progress: 70% - Starting voice synthesis")
        
        tts_start = time.time()
        unique_chords = list(set(chord_data['speech'] for chord_data in chords))
        print(f"[TASK {task_id}] Unique chords to synthesize: {len(unique_chords)}")
        print(f"[TASK {task_id}] Unique chords: {unique_chords}")
        
        tts_cache = {}
        
        # Update progress for each chord synthesis
        for i, chord_speech in enumerate(unique_chords):
            # Update progress for each chord (70-85%) with whole numbers only
            # Simple mapping: 0->70%, 1->72%, 2->75%, 3->77%, 4->80%, 5->82%, 6->85%
            if len(unique_chords) == 1:
                chord_progress = 70
            elif len(unique_chords) == 2:
                chord_progress = 70 if i == 0 else 85
            elif len(unique_chords) == 3:
                chord_progress = 70 if i == 0 else (77 if i == 1 else 85)
            elif len(unique_chords) == 4:
                chord_progress = 70 if i == 0 else (75 if i == 1 else (80 if i == 2 else 85))
            else:
                # For 5+ chords, use simple increments
                chord_progress = 70 + (i * 3)  # 70, 73, 76, 79, 82, 85
                if chord_progress > 85:
                    chord_progress = 85
            
            tasks[task_id]['progress'] = chord_progress
            tasks[task_id]['step'] = f'Synthesizing chord {i+1}/{len(unique_chords)}'
            print(f"[TASK {task_id}] Progress: {chord_progress}% - Synthesizing chord {i+1}/{len(unique_chords)}: {chord_speech}")
            
            tts_output_path = os.path.join(task_dir, f'tts_{chord_speech.replace(" ", "_").replace("#", "sharp")}.wav')
            if not synthesize_chord_speech(chord_speech, voice_sample_path, tts_output_path):
                error_msg = f"TTS synthesis failed for chord: {chord_speech}"
                print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)
            if os.path.exists(tts_output_path):
                from pydub import AudioSegment
                tts_cache[chord_speech] = AudioSegment.from_wav(tts_output_path)
                print(f"Loaded TTS for '{chord_speech}': {len(tts_cache[chord_speech])}ms duration")
            else:
                error_msg = f"TTS output file not created for chord: {chord_speech}"
                print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)
        
        # Step 6: Creating chord audio track
        print(f"Step 6: Creating chord audio track for task {task_id}")
        tasks[task_id]['step'] = 'Creating chord audio track'
        tasks[task_id]['progress'] = 85
        print(f"[TASK {task_id}] Progress: 85% - Creating chord audio track")
        chord_audio_segments = []
        for i, chord_data in enumerate(chords):
            if i == 0:
                silence_duration = chord_data['time'] * 1000
            else:
                silence_duration = (chord_data['time'] - chords[i-1]['time']) * 1000
            if silence_duration > 0:
                chord_audio_segments.append(AudioSegment.silent(duration=int(silence_duration)))
                print(f"Added {int(silence_duration)}ms silence before chord {i+1}")
            chord_speech = chord_data['speech']
            if chord_speech in tts_cache:
                speech_audio = tts_cache[chord_speech]
                chord_audio_segments.append(speech_audio)
                print(f"Added '{chord_speech}' at {chord_data['time']:.2f}s: {len(speech_audio)}ms duration")
            else:
                beep = AudioSegment.sine(frequency=440, duration=200)
                chord_audio_segments.append(beep)
                print(f"Added fallback beep for '{chord_speech}' at {chord_data['time']:.2f}s")
        chord_track = sum(chord_audio_segments, AudioSegment.empty())
        
        # Step 7: Mixing final audio
        print(f"Step 7: Mixing final audio for task {task_id}")
        tasks[task_id]['step'] = 'Overlaying spoken chords onto instrumental track'
        tasks[task_id]['progress'] = 90
        print(f"[TASK {task_id}] Progress: 90% - Mixing final audio")
        instrumental_audio = AudioSegment.from_wav(instrumental_path)
        if len(chord_track) < len(instrumental_audio):
            chord_track += AudioSegment.silent(duration=len(instrumental_audio) - len(chord_track))
        elif len(chord_track) > len(instrumental_audio):
            chord_track = chord_track[:len(instrumental_audio)]
        final_audio = instrumental_audio.overlay(chord_track - 10)
        output_path = os.path.join(task_dir, 'final.mp3')
        final_audio.export(output_path, format='mp3')
        
        # Complete
        total_time = time.time() - start_time
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['step'] = 'Complete'
        tasks[task_id]['progress'] = 100
        tasks[task_id]['output_file'] = output_path
        print(f"\n=== [TASK {task_id}] PROCESSING COMPLETED ===")
        print(f"[TASK {task_id}] Progress: 100% - Processing completed successfully")
        print(f"[TASK {task_id}] Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"[TASK {task_id}] Output file: {output_path}")
        
    except Exception as e:
        print(f"\n=== [TASK {task_id}] PROCESSING ERROR ===")
        print(f"[TASK {task_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)
        tasks[task_id]['step'] = f'Error: {str(e)}'
        print(f"[TASK {task_id}] Processing error for task {task_id}: {e}")

@app.route('/')
def index():
    """Serve the main application page"""
    try:
        return send_file('index.html')
    except Exception as e:
        print(f"Error serving index.html: {e}")
        return f"Error: {str(e)}", 500

@app.route('/Logo.png')
def serve_logo():
    """Serve the logo image"""
    try:
        # Try the main logo file first
        if os.path.exists('Logo-transparent.png'):
            return send_file('Logo-transparent.png', mimetype='image/png')
        elif os.path.exists('Logo-transparent-final.png'):
            return send_file('Logo-transparent-final.png', mimetype='image/png')
        elif os.path.exists('Logo.png'):
            return send_file('Logo.png', mimetype='image/png')
        else:
            print("No logo file found")
            return "Logo not found", 404
    except Exception as e:
        print(f"Error serving Logo-transparent.png: {e}")
        return f"Error: {str(e)}", 500

@app.route('/Logo-transparent.png')
def serve_logo_transparent():
    """Serve the transparent logo image"""
    try:
        # Try the main logo file first
        if os.path.exists('Logo-transparent.png'):
            return send_file('Logo-transparent.png', mimetype='image/png')
        elif os.path.exists('Logo-transparent-final.png'):
            return send_file('Logo-transparent-final.png', mimetype='image/png')
        elif os.path.exists('Logo-transparent-inverted.png'):
            return send_file('Logo-transparent-inverted.png', mimetype='image/png')
        elif os.path.exists('Logo.png'):
            return send_file('Logo.png', mimetype='image/png')
        else:
            print("No logo file found")
            return "Logo not found", 404
    except Exception as e:
        print(f"Error serving Logo-transparent.png: {e}")
        return f"Error: {str(e)}", 500

@app.route('/favicon.ico')
def serve_favicon_ico():
    """Serve the favicon.ico file"""
    try:
        response = send_file('favicon.ico', mimetype='image/x-icon')
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    except Exception as e:
        print(f"Error serving favicon.ico: {e}")
        return f"Error: {str(e)}", 500

@app.route('/Logo-transparent-inverted.png')
def serve_logo_transparent_inverted():
    """Serve the inverted transparent logo image"""
    try:
        return send_file('Logo-transparent-inverted.png', mimetype='image/png')
    except Exception as e:
        print(f"Error serving Logo-transparent-inverted.png: {e}")
        return f"Error: {str(e)}", 500

@app.route('/favicon.png')
def serve_favicon_png():
    """Serve the favicon.png file"""
    try:
        return send_file('favicon.png', mimetype='image/png')
    except Exception as e:
        print(f"Error serving favicon.png: {e}")
        return f"Error: {str(e)}", 500

@app.route('/test')
def test():
    """Simple test route to verify the app is working"""
    return jsonify({
        'status': 'ok',
        'message': 'Flask app is running',
        'version': VERSION
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Remove chord type selections (always use madmom's default)
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    task_dir = os.path.join(UPLOAD_FOLDER, task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(task_dir, filename)
    file.save(file_path)
    
    # Initialize task
    tasks[task_id] = {
        'status': 'queued',
        'step': 'Uploaded',
        'filename': filename
    }
    
    # Start background processing
    thread = Thread(target=process_audio_task, args=(task_id, file_path))
    thread.start()
    
    return jsonify({'task_id': task_id, 'status': 'queued'})

@app.route('/status/<task_id>')
def get_status(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    # Don't log status requests to reduce terminal noise
    return jsonify(tasks[task_id])

@app.route('/cancel/<task_id>', methods=['POST'])
def cancel_task(task_id):
    """Cancel a running task"""
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = tasks[task_id]
    if task['status'] not in ['processing', 'queued']:
        return jsonify({'error': 'Task cannot be cancelled'}), 400
    
    # Mark task as cancelled
    task['status'] = 'cancelled'
    task['step'] = 'Cancelled by user'
    
    return jsonify({'status': 'cancelled', 'message': 'Task cancelled successfully'})

@app.route('/download/<task_id>')
def download_file(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = tasks[task_id]
    if task['status'] != 'completed':
        return jsonify({'error': 'Task not completed'}), 400
    
    return send_file(task['output_file'], as_attachment=True, download_name='chord_vocals.mp3')

@app.route('/chords/<task_id>')
def get_chords(task_id):
    """Get chord progression data for a completed task"""
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = tasks[task_id]
    if task['status'] != 'completed':
        return jsonify({'error': 'Task not completed'}), 400
    
    # Load chord data from the saved JSON file
    chords_file = os.path.join(UPLOAD_FOLDER, task_id, 'chords.json')
    if not os.path.exists(chords_file):
        return jsonify({'error': 'Chord data not found'}), 404
    
    try:
        with open(chords_file, 'r') as f:
            chords = json.load(f)
        return jsonify({'chords': chords})
    except Exception as e:
        return jsonify({'error': f'Failed to load chord data: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    # GPU detection for health check
    gpu_info = {}
    try:
        import torch
        gpu_info['pytorch_version'] = torch.__version__
        gpu_info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            gpu_info['cuda_device_count'] = torch.cuda.device_count()
            gpu_info['cuda_device_name'] = torch.cuda.get_device_name()
            gpu_info['cuda_memory_allocated_gb'] = round(torch.cuda.memory_allocated() / 1024**3, 2)
        else:
            gpu_info['cuda_device_count'] = 0
            gpu_info['cuda_device_name'] = 'None'
            gpu_info['cuda_memory_allocated_gb'] = 0
    except ImportError:
        gpu_info['pytorch_version'] = 'Not available'
        gpu_info['cuda_available'] = False
        gpu_info['cuda_device_count'] = 0
        gpu_info['cuda_device_name'] = 'None'
        gpu_info['cuda_memory_allocated_gb'] = 0
    
    return jsonify({
        'status': 'healthy',
        'version': VERSION,
        'name': 'ChordiSpeak',
        'gpu': gpu_info
    })

@app.route('/docs')
def api_docs():
    """Serve API documentation page"""
    docs_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ChordiSpeak API Documentation</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
                    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
                    sans-serif;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }

            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 2rem;
            }

            .header {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                padding: 2rem;
                margin-bottom: 2rem;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
                text-align: center;
            }

            .header h1 {
                font-size: 3rem;
                color: #667eea;
                margin-bottom: 1rem;
            }

            .header p {
                font-size: 1.2rem;
                color: #666;
                max-width: 600px;
                margin: 0 auto;
            }

            .endpoint {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                padding: 2rem;
                margin-bottom: 2rem;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            }

            .endpoint h2 {
                color: #667eea;
                margin-bottom: 1rem;
                font-size: 1.8rem;
            }

            .method {
                display: inline-block;
                padding: 0.5rem 1rem;
                border-radius: 8px;
                font-weight: bold;
                margin-right: 1rem;
            }

            .method.post { background: #4CAF50; color: white; }
            .method.get { background: #2196F3; color: white; }

            .url {
                font-family: 'Courier New', monospace;
                background: #f5f5f5;
                padding: 0.5rem;
                border-radius: 5px;
                margin: 1rem 0;
                display: inline-block;
            }

            .description {
                margin: 1rem 0;
                line-height: 1.6;
            }

            .params {
                background: #f8f9ff;
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
            }

            .params h4 {
                color: #667eea;
                margin-bottom: 0.5rem;
            }

            .param {
                margin: 0.5rem 0;
                padding: 0.5rem;
                background: white;
                border-radius: 5px;
                border-left: 4px solid #667eea;
            }

            .response {
                background: #f0f8ff;
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
            }

            .response h4 {
                color: #667eea;
                margin-bottom: 0.5rem;
            }

            .example {
                background: #f5f5f5;
                padding: 1rem;
                border-radius: 5px;
                font-family: 'Courier New', monospace;
                margin: 0.5rem 0;
                overflow-x: auto;
            }

            .back-link {
                display: inline-block;
                margin-bottom: 2rem;
                padding: 0.5rem 1rem;
                background: rgba(255, 255, 255, 0.9);
                border-radius: 10px;
                text-decoration: none;
                color: #667eea;
                font-weight: bold;
                transition: all 0.3s ease;
            }

            .back-link:hover {
                background: white;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            }

            .status-codes {
                margin: 1rem 0;
            }

            .status-code {
                display: inline-block;
                padding: 0.25rem 0.5rem;
                border-radius: 5px;
                font-size: 0.9rem;
                margin: 0.25rem;
            }

            .status-200 { background: #4CAF50; color: white; }
            .status-400 { background: #FF9800; color: white; }
            .status-404 { background: #F44336; color: white; }
            .status-500 { background: #9C27B0; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/" class="back-link">← Back to ChordiSpeak</a>
            
            <div class="header">
                <h1>🎵 ChordiSpeak API</h1>
                <p>RESTful API for AI-powered chord vocal generation</p>
            </div>

            <div class="endpoint">
                <h2><span class="method post">POST</span> Upload Audio File</h2>
                <div class="url">/upload</div>
                <div class="description">
                    Upload an audio file for chord detection and vocal synthesis processing.
                </div>
                
                <div class="params">
                    <h4>Parameters</h4>
                    <div class="param">
                        <strong>file</strong> (multipart/form-data) - Audio file (MP3, WAV, FLAC, M4A)
                    </div>
                </div>

                <div class="response">
                    <h4>Response</h4>
                    <div class="status-codes">
                        <span class="status-code status-200">200 OK</span>
                        <span class="status-code status-400">400 Bad Request</span>
                    </div>
                    <div class="example">
{
  "task_id": "uuid-string",
  "message": "File uploaded successfully",
  "filename": "song.mp3"
}
                    </div>
                </div>
            </div>

            <div class="endpoint">
                <h2><span class="method get">GET</span> Check Processing Status</h2>
                <div class="url">/status/{task_id}</div>
                <div class="description">
                    Check the status of a processing task and get progress updates.
                </div>
                
                <div class="params">
                    <h4>Parameters</h4>
                    <div class="param">
                        <strong>task_id</strong> (path) - Task identifier from upload response
                    </div>
                </div>

                <div class="response">
                    <h4>Response</h4>
                    <div class="status-codes">
                        <span class="status-code status-200">200 OK</span>
                        <span class="status-code status-404">404 Not Found</span>
                    </div>
                    <div class="example">
{
  "status": "processing",
  "progress": 75,
  "step": "Synthesizing vocals",
  "message": "Processing your audio..."
}
                    </div>
                </div>
            </div>

            <div class="endpoint">
                <h2><span class="method get">GET</span> Download Result</h2>
                <div class="url">/download/{task_id}</div>
                <div class="description">
                    Download the processed audio file once processing is complete.
                </div>
                
                <div class="params">
                    <h4>Parameters</h4>
                    <div class="param">
                        <strong>task_id</strong> (path) - Task identifier from upload response
                    </div>
                </div>

                <div class="response">
                    <h4>Response</h4>
                    <div class="status-codes">
                        <span class="status-code status-200">200 OK</span>
                        <span class="status-code status-404">404 Not Found</span>
                    </div>
                    <div class="description">
                        Returns the processed audio file as a downloadable WAV file.
                    </div>
                </div>
            </div>

            <div class="endpoint">
                <h2><span class="method get">GET</span> Health Check</h2>
                <div class="url">/health</div>
                <div class="description">
                    Check if the API is running and healthy.
                </div>

                <div class="response">
                    <h4>Response</h4>
                    <div class="status-codes">
                        <span class="status-code status-200">200 OK</span>
                    </div>
                    <div class="example">
{
  "status": "healthy",
  "service": "chordispeak"
}
                    </div>
                </div>
            </div>

            <div class="endpoint">
                <h2>Processing Steps</h2>
                <div class="description">
                    The audio processing pipeline includes the following steps:
                </div>
                <div class="params">
                    <div class="param">1. <strong>Vocal Separation</strong> - Extract vocals from instrumental using Demucs AI. Output files: <code>vocal_track.wav</code> (vocals only), <code>instrumental_track.wav</code> (instrumental only).</div>
                    <div class="param">2. <strong>Voice Sample Extraction</strong> - Extract clean voice sample from separated vocals</div>
                    <div class="param">3. <strong>Chord Detection</strong> - Analyze instrumental track to detect chord progressions</div>
                    <div class="param">4. <strong>Voice Cloning</strong> - Use voice sample for voice cloning in TTS</div>
                    <div class="param">5. <strong>Speech Synthesis</strong> - Generate chord vocals using voice-cloned TTS</div>
                    <div class="param">6. <strong>Audio Mixing</strong> - Overlay synthesized chord vocals onto instrumental track</div>
                </div>
            </div>

            <div class="endpoint">
                <h2>Supported Audio Formats</h2>
                <div class="description">
                    The API supports the following audio formats:
                </div>
                <div class="params">
                    <div class="param">• MP3 (.mp3)</div>
                    <div class="param">• WAV (.wav)</div>
                    <div class="param">• FLAC (.flac)</div>
                    <div class="param">• M4A (.m4a)</div>
                </div>
                <div class="description">
                    <strong>Maximum file size:</strong> 50MB
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return docs_html

@app.route('/debug/<task_id>')
def debug_task(task_id):
    """Debug endpoint to see task details"""
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = tasks[task_id]
    logs = task_logs.get(task_id, [])
    return jsonify({
        'task_id': task_id,
        'status': task.get('status', 'unknown'),
        'step': task.get('step', 'unknown'),
        'progress': task.get('progress', 0),
        'demucs_percentage': task.get('demucs_percentage', 0),
        'error': task.get('error', None),
        'filename': task.get('filename', 'unknown'),
        'output_file': task.get('output_file', None),
        'logs': logs
    })

@app.route('/logs/<task_id>')
def get_task_logs(task_id):
    """Get debug logs for a specific task"""
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    logs = task_logs.get(task_id, [])
    return jsonify({
        'task_id': task_id,
        'logs': logs
    })

@app.route('/gpu-info')
def get_gpu_info():
    """Get detailed GPU information for debugging"""
    gpu_info = {}
    try:
        import torch
        gpu_info['pytorch_version'] = torch.__version__
        gpu_info['cuda_available'] = torch.cuda.is_available()
        gpu_info['cuda_version'] = torch.version.cuda if torch.cuda.is_available() else 'None'
        
        if torch.cuda.is_available():
            gpu_info['cuda_device_count'] = torch.cuda.device_count()
            gpu_info['cuda_current_device'] = torch.cuda.current_device()
            gpu_info['cuda_device_name'] = torch.cuda.get_device_name()
            gpu_info['cuda_memory_allocated_gb'] = round(torch.cuda.memory_allocated() / 1024**3, 2)
            gpu_info['cuda_memory_reserved_gb'] = round(torch.cuda.memory_reserved() / 1024**3, 2)
            gpu_info['cuda_memory_cached_gb'] = round(torch.cuda.memory_reserved() / 1024**3, 2)
            
            # Get GPU properties
            props = torch.cuda.get_device_properties(0)
            gpu_info['gpu_name'] = props.name
            gpu_info['gpu_memory_total_gb'] = round(props.total_memory / 1024**3, 2)
            gpu_info['gpu_memory_free_gb'] = round((props.total_memory - torch.cuda.memory_allocated()) / 1024**3, 2)
        else:
            gpu_info['cuda_device_count'] = 0
            gpu_info['cuda_current_device'] = -1
            gpu_info['cuda_device_name'] = 'None'
            gpu_info['cuda_memory_allocated_gb'] = 0
            gpu_info['cuda_memory_reserved_gb'] = 0
            gpu_info['cuda_memory_cached_gb'] = 0
            gpu_info['gpu_name'] = 'None'
            gpu_info['gpu_memory_total_gb'] = 0
            gpu_info['gpu_memory_free_gb'] = 0
    except ImportError:
        gpu_info['pytorch_version'] = 'Not available'
        gpu_info['cuda_available'] = False
        gpu_info['cuda_version'] = 'None'
        gpu_info['cuda_device_count'] = 0
        gpu_info['cuda_current_device'] = -1
        gpu_info['cuda_device_name'] = 'None'
        gpu_info['cuda_memory_allocated_gb'] = 0
        gpu_info['cuda_memory_reserved_gb'] = 0
        gpu_info['cuda_memory_cached_gb'] = 0
        gpu_info['gpu_name'] = 'None'
        gpu_info['gpu_memory_total_gb'] = 0
        gpu_info['gpu_memory_free_gb'] = 0
    
    return jsonify(gpu_info)

def process_audio_task(task_id, file_path):
    """Process uploaded audio file through the complete pipeline"""
    print(f"\n=== [TASK {task_id}] STARTING PROCESSING ===")
    print(f"[TASK {task_id}] Processing file: {file_path}")
    
    start_time = time.time()
    
    try:
        # Update task status
        tasks[task_id]['status'] = 'processing'
        tasks[task_id]['step'] = 'Preparing audio file'
        tasks[task_id]['progress'] = 5
        print(f"[TASK {task_id}] Progress: 5% - Starting processing")
        
        # Step 1: Audio preparation
        print(f"\n=== [TASK {task_id}] STEP 1: AUDIO PREPARATION ===")
        tasks[task_id]['step'] = 'Preparing audio file'
        tasks[task_id]['progress'] = 10
        print(f"[TASK {task_id}] Progress: 10% - Audio preparation")
        
        # Convert to WAV if needed
        input_path = file_path
        if not file_path.lower().endswith('.wav'):
            from pydub import AudioSegment
            audio = AudioSegment.from_file(file_path)
            input_path = os.path.join(os.path.dirname(file_path), 'input.wav')
            audio.export(input_path, format='wav')
            print(f"[TASK {task_id}] Converted to WAV: {input_path}")
        
        # Step 2: Vocal separation with Demucs
        print(f"\n=== [TASK {task_id}] STEP 2: VOCAL SEPARATION ===")
        tasks[task_id]['step'] = 'Splitting vocal & instrumental'
        tasks[task_id]['progress'] = 15
        print(f"[TASK {task_id}] Progress: 15% - Starting vocal separation")
        
        task_dir = os.path.dirname(input_path)
        
        # Run Demucs separation
        demucs_cmd = [
            'demucs', '--two-stems=vocals', '--out', task_dir, 
            '--mp3', '--mp3-bitrate', '128', input_path
        ]
        
        print(f"[TASK {task_id}] Running Demucs: {' '.join(demucs_cmd)}")
        process = subprocess.Popen(
            demucs_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor Demucs progress
        demucs_output = []
        for line in process.stdout:
            demucs_output.append(line.strip())
            if 'progress' in line.lower() or '%' in line:
                print(f"[TASK {task_id}] Demucs: {line.strip()}")
        
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError(f"Demucs failed with return code {process.returncode}")
        
        print(f"[TASK {task_id}] Demucs separation completed successfully")
        
        # Update progress
        tasks[task_id]['step'] = 'Vocal separation complete'
        tasks[task_id]['progress'] = 30
        print(f"[TASK {task_id}] Progress: 30% - Vocal separation complete")
        
        # Step 3: Voice sample extraction
        print(f"\n=== [TASK {task_id}] STEP 3: VOICE SAMPLE EXTRACTION ===")
        tasks[task_id]['step'] = 'Extracting voice sample'
        tasks[task_id]['progress'] = 35
        print(f"[TASK {task_id}] Progress: 35% - Extracting voice sample")
        
        # Find Demucs output files
        demucs_output_dir = os.path.join(task_dir, 'htdemucs')
        if not os.path.exists(demucs_output_dir):
            raise RuntimeError("Demucs output directory not found")
        
        # Look for the input subdirectory
        input_subdir = None
        for item in os.listdir(demucs_output_dir):
            item_path = os.path.join(demucs_output_dir, item)
            if os.path.isdir(item_path):
                input_subdir = item_path
                break
        
        if not input_subdir:
            raise RuntimeError("Demucs input subdirectory not found")
        
        # Define paths for vocal and instrumental files
        vocal_path = os.path.join(input_subdir, 'vocals.mp3')
        instrumental_path = os.path.join(input_subdir, 'no_vocals.mp3')
        
        voice_sample_path = os.path.join(task_dir, 'voice_sample.wav')
        
        # Check if vocal file exists
        if not os.path.exists(vocal_path):
            raise RuntimeError(f"Vocal track not found after Demucs separation. Expected: {vocal_path}")
        
        # Convert MP3 to WAV for voice cloning
        from pydub import AudioSegment
        vocal_audio = AudioSegment.from_mp3(vocal_path)
        vocal_audio.export(voice_sample_path, format='wav')
        print(f"[TASK {task_id}] Voice sample extracted: {voice_sample_path}")
        
        # Step 4: Chord detection
        print(f"\n=== [TASK {task_id}] STEP 4: CHORD DETECTION ===")
        tasks[task_id]['step'] = 'Analyzing chord pattern'
        tasks[task_id]['progress'] = 40
        print(f"[TASK {task_id}] Progress: 40% - Starting chord detection")
        
        # Check if instrumental file exists
        if not os.path.exists(instrumental_path):
            raise RuntimeError(f"Instrumental track not found after Demucs separation. Expected: {instrumental_path}")
        
        # Convert MP3 to WAV for chord detection
        instrumental_wav_path = os.path.join(task_dir, 'instrumental_track.wav')
        from pydub import AudioSegment
        instrumental_audio = AudioSegment.from_mp3(instrumental_path)
        instrumental_audio.export(instrumental_wav_path, format='wav')
        print(f"[TASK {task_id}] Converted instrumental to WAV: {instrumental_wav_path}")
        
        # Use the madmom compatibility layer
        from madmom_compat import detect_chords_simple
        chords = detect_chords_simple(instrumental_wav_path, task_id)
        
        if not chords or len(chords) == 0:
            raise RuntimeError("No chords detected. This could indicate an issue with the audio file or chord detection.")
        
        print(f"[TASK {task_id}] Raw madmom output: {len(chords)} chord segments")
        
        # Process madmom output format
        valid_chords = []
        filtered_out_count = 0
        min_confidence = 0.5
        min_chord_duration = 0.5
        min_time_between_chords = 1.0
        
        print(f"[TASK {task_id}] Processing {len(chords)} chord segments from madmom...")
        
        for i, chord_data in enumerate(chords):
            try:
                # Madmom format: [start_time, end_time, chord_label, confidence]
                if len(chord_data) >= 4:
                    start_time = float(chord_data[0])
                    end_time = float(chord_data[1])
                    chord_label = str(chord_data[2])
                    confidence = float(chord_data[3])
                elif len(chord_data) == 3:
                    start_time = float(chord_data[0])
                    end_time = float(chord_data[1])
                    chord_label = str(chord_data[2])
                    confidence = 1.0  # Default confidence
                else:
                    print(f"[TASK {task_id}] Skipping invalid chord data format: {chord_data}")
                    filtered_out_count += 1
                    continue
                
                # Skip 'N' (no chord) detections
                if chord_label == 'N':
                    filtered_out_count += 1
                    continue
                
                # Filter by confidence threshold
                if confidence < min_confidence:
                    print(f"[TASK {task_id}] Filtered out low confidence chord: {chord_label} at {start_time:.2f}s (confidence: {confidence:.3f} < {min_confidence})")
                    filtered_out_count += 1
                    continue
                
                # Calculate chord duration
                chord_duration = end_time - start_time
                
                # Skip very short chord detections
                if chord_duration < min_chord_duration:
                    print(f"[TASK {task_id}] Filtered out short chord: {chord_label} at {start_time:.2f}s (duration: {chord_duration:.3f}s < {min_chord_duration}s)")
                    filtered_out_count += 1
                    continue
                
                # Convert madmom chord labels to our format
                chord_name = convert_madmom_chord(chord_label)
                
                print(f"[TASK {task_id}] Processing chord: {chord_label} -> {chord_name} at {start_time:.2f}-{end_time:.2f}s (confidence: {confidence:.3f})")
                
                valid_chords.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'chord': chord_name,
                    'confidence': confidence,
                    'duration': chord_duration
                })
                
            except Exception as e:
                print(f"[TASK {task_id}] Error processing chord data {chord_data}: {e}")
                filtered_out_count += 1
                continue
        
        print(f"[TASK {task_id}] Successfully processed {len(valid_chords)} chord segments (filtered out {filtered_out_count})")
        
        if not valid_chords:
            raise RuntimeError("No valid chords detected. This could indicate an issue with the audio file or chord detection.")
        
        # Sort by start time
        valid_chords.sort(key=lambda x: x['start_time'])
        
        # Apply smoothing and timing optimization for speech synthesis
        print(f"[TASK {task_id}] Applying chord smoothing and timing optimization...")
        
        # Apply median filtering to reduce rapid switching
        window_size = 2
        smoothed_chords = []
        
        for i in range(len(valid_chords)):
            # Get window of chords around current position
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(valid_chords), i + window_size // 2 + 1)
            window = valid_chords[start_idx:end_idx]
            
            # Find most common chord in window
            chord_counts = {}
            for chord_info in window:
                chord = chord_info['chord']
                chord_counts[chord] = chord_counts.get(chord, 0) + 1
            
            # Use the most common chord in the window
            most_common_chord = max(chord_counts.items(), key=lambda x: x[1])[0]
            
            # Only add if it's different from the last added chord
            if not smoothed_chords or most_common_chord != smoothed_chords[-1]['chord']:
                smoothed_chords.append({
                    'time': valid_chords[i]['start_time'],
                    'chord': most_common_chord,
                    'speech': chord_to_ipa_phonemes(most_common_chord),
                    'confidence': valid_chords[i]['confidence'],
                    'duration': valid_chords[i]['duration']
                })
        
        # Final pass: ensure minimum time between chord changes
        final_chords = []
        for chord_info in smoothed_chords:
            if not final_chords or (chord_info['time'] - final_chords[-1]['time']) >= min_time_between_chords:
                final_chords.append(chord_info)
            else:
                print(f"[TASK {task_id}] Filtered out rapid chord change: {chord_info['chord']} at {chord_info['time']:.2f}s (too close to previous)")
        
        print(f"[TASK {task_id}] Final result: {len(final_chords)} chord changes")
        for chord_info in final_chords:
            print(f"[TASK {task_id}]   {chord_info['chord']} at {chord_info['time']:.2f}s")
        
        # Update progress: Chord detection complete (65%)
        if task_id and task_id in tasks:
            tasks[task_id]['step'] = 'Analyzing chord pattern (complete)'
            tasks[task_id]['progress'] = 65
            print(f"[TASK {task_id}] Progress: 65% - Chord detection and processing complete")
        
        # Save chord data
        chords_file = os.path.join(task_dir, 'chords.json')
        with open(chords_file, 'w') as f:
            json.dump(final_chords, f)
        print(f"Chord data saved: {chords_file}")
        
        # Step 5: Voice synthesis using voice cloning
        print(f"\n=== [TASK {task_id}] STEP 5: VOICE SYNTHESIS ===")
        tasks[task_id]['step'] = 'Synthesizing spoken chord overlay'
        tasks[task_id]['progress'] = 70
        print(f"[TASK {task_id}] Progress: 70% - Starting voice synthesis")
        
        tts_start = time.time()
        unique_chords = list(set(chord_data['speech'] for chord_data in final_chords))
        print(f"[TASK {task_id}] Unique chords to synthesize: {len(unique_chords)}")
        print(f"[TASK {task_id}] Unique chords: {unique_chords}")
        
        tts_cache = {}
        
        # Update progress for each chord synthesis
        for i, chord_speech in enumerate(unique_chords):
            # Update progress for each chord (70-85%) with whole numbers only
            if len(unique_chords) == 1:
                chord_progress = 70
            elif len(unique_chords) == 2:
                chord_progress = 70 if i == 0 else 85
            elif len(unique_chords) == 3:
                chord_progress = 70 if i == 0 else (77 if i == 1 else 85)
            elif len(unique_chords) == 4:
                chord_progress = 70 if i == 0 else (75 if i == 1 else (80 if i == 2 else 85))
            else:
                # For 5+ chords, use simple increments
                chord_progress = 70 + (i * 3)  # 70, 73, 76, 79, 82, 85
                if chord_progress > 85:
                    chord_progress = 85
            
            tasks[task_id]['progress'] = chord_progress
            tasks[task_id]['step'] = f'Synthesizing chord {i+1}/{len(unique_chords)}'
            print(f"[TASK {task_id}] Progress: {chord_progress}% - Synthesizing chord {i+1}/{len(unique_chords)}: {chord_speech}")
            
            tts_output_path = os.path.join(task_dir, f'tts_{chord_speech.replace(" ", "_").replace("#", "sharp")}.wav')
            if not synthesize_chord_speech(chord_speech, voice_sample_path, tts_output_path):
                error_msg = f"TTS synthesis failed for chord: {chord_speech}"
                print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)
            if os.path.exists(tts_output_path):
                tts_cache[chord_speech] = AudioSegment.from_wav(tts_output_path)
                print(f"Loaded TTS for '{chord_speech}': {len(tts_cache[chord_speech])}ms duration")
            else:
                error_msg = f"TTS output file not created for chord: {chord_speech}"
                print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)
        
        # Step 6: Creating chord audio track
        print(f"Step 6: Creating chord audio track for task {task_id}")
        tasks[task_id]['step'] = 'Creating chord audio track'
        tasks[task_id]['progress'] = 85
        print(f"[TASK {task_id}] Progress: 85% - Creating chord audio track")
        from pydub import AudioSegment
        chord_audio_segments = []
        for i, chord_data in enumerate(final_chords):
            if i == 0:
                silence_duration = chord_data['time'] * 1000
            else:
                silence_duration = (chord_data['time'] - final_chords[i-1]['time']) * 1000
            if silence_duration > 0:
                chord_audio_segments.append(AudioSegment.silent(duration=int(silence_duration)))
                print(f"Added {int(silence_duration)}ms silence before chord {i+1}")
            chord_speech = chord_data['speech']
            if chord_speech in tts_cache:
                speech_audio = tts_cache[chord_speech]
                chord_audio_segments.append(speech_audio)
                print(f"Added '{chord_speech}' at {chord_data['time']:.2f}s: {len(speech_audio)}ms duration")
            else:
                beep = AudioSegment.sine(frequency=440, duration=200)
                chord_audio_segments.append(beep)
                print(f"Added fallback beep for '{chord_speech}' at {chord_data['time']:.2f}s")
        chord_track = sum(chord_audio_segments, AudioSegment.empty())
        
        # Step 7: Mixing final audio
        print(f"Step 7: Mixing final audio for task {task_id}")
        tasks[task_id]['step'] = 'Overlaying spoken chords onto instrumental track'
        tasks[task_id]['progress'] = 90
        print(f"[TASK {task_id}] Progress: 90% - Mixing final audio")
        instrumental_audio = AudioSegment.from_wav(instrumental_path)
        if len(chord_track) < len(instrumental_audio):
            chord_track += AudioSegment.silent(duration=len(instrumental_audio) - len(chord_track))
        elif len(chord_track) > len(instrumental_audio):
            chord_track = chord_track[:len(instrumental_audio)]
        final_audio = instrumental_audio.overlay(chord_track - 10)
        output_path = os.path.join(task_dir, 'final.mp3')
        final_audio.export(output_path, format='mp3')
        
        # Complete
        total_time = time.time() - start_time
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['step'] = 'Complete'
        tasks[task_id]['progress'] = 100
        tasks[task_id]['output_file'] = output_path
        print(f"\n=== [TASK {task_id}] PROCESSING COMPLETED ===")
        print(f"[TASK {task_id}] Progress: 100% - Processing completed successfully")
        print(f"[TASK {task_id}] Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"[TASK {task_id}] Output file: {output_path}")
        
    except Exception as e:
        print(f"\n=== [TASK {task_id}] PROCESSING ERROR ===")
        print(f"[TASK {task_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)
        tasks[task_id]['step'] = f'Error: {str(e)}'
        print(f"[TASK {task_id}] Processing error for task {task_id}: {e}")

def test_pronunciation_strategies():
    """Test different pronunciation strategies for letter names"""
    strategies = {
        'IPA': {
            'A': 'ˈeɪ', 'B': 'ˈbiː', 'C': 'ˈsiː', 'D': 'ˈdiː', 'E': 'ˈiː', 'F': 'ˈɛf', 'G': 'ˈdʒiː'
        },
        'Dots': {
            'A': 'A dot', 'B': 'B dot', 'C': 'C dot', 'D': 'D dot', 'E': 'E dot', 'F': 'F dot', 'G': 'G dot'
        },
        'Words': {
            'A': 'A as in apple', 'B': 'B as in boy', 'C': 'C as in cat', 'D': 'D as in dog', 
            'E': 'E as in easy', 'F': 'F as in fun', 'G': 'G as in go'
        },
        'NATO': {
            'A': 'Alpha', 'B': 'Bravo', 'C': 'Charlie', 'D': 'Delta', 'E': 'Echo', 'F': 'Foxtrot', 'G': 'Golf'
        },
        'Simple': {
            'A': 'AY', 'B': 'BEE', 'C': 'SEE', 'D': 'DEE', 'E': 'EE', 'F': 'EFF', 'G': 'GEE'
        }
    }
    
    print("Testing pronunciation strategies:")
    for strategy_name, letters in strategies.items():
        print(f"\n{strategy_name}:")
        for letter, pronunciation in letters.items():
            print(f"  {letter} → {pronunciation}")
    
    return strategies

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    # Disable Flask's default request logging to reduce terminal noise
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(debug=False, host='0.0.0.0', port=port)
