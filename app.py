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
        chord_detector = DeepChromaChordRecognitionProcessor()
        print(f"[TASK {task_id}] Madmom chord detector initialized successfully")
        
        # Update progress: Starting chord detection (40-50%)
        if task_id and task_id in tasks:
            tasks[task_id]['step'] = 'Analyzing chord pattern'
            tasks[task_id]['progress'] = 40
            print(f"[TASK {task_id}] Progress: 40% - Starting chord detection")
        
        # Process the audio file with the chord detector
        print(f"[TASK {task_id}] Starting chord detection...")
        print(f"[TASK {task_id}] Audio file path: {audio_file}")
        
        # Use madmom's native chord detection - pass file path directly
        chords = chord_detector(audio_file)
        print(f"[TASK {task_id}] Chord detection completed successfully")
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
