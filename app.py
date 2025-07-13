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

def extract_voice_sample(vocals_path, sample_duration=None):
    """Extract voice sample from separated vocals for voice cloning"""
    if not lazy_import_audio_deps():
        print("Audio processing dependencies not available")
        return None, None
        
    try:
        # Use the full vocal track for better voice cloning
        y, sr = librosa.load(vocals_path)
        
        # Return the entire vocal track
        return y, sr
        
    except Exception as e:
        print(f"Voice sample extraction error: {e}")
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
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        
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
    """Detect chords from audio file using madmom for accurate chord recognition with progress tracking"""
    if not lazy_import_audio_deps():
        raise RuntimeError("Audio processing dependencies not available. Cannot perform chord detection.")
        
    try:
        import time
        from madmom.audio.chroma import DeepChromaProcessor
        from madmom.features.chords import DeepChromaChordRecognitionProcessor
        
        # Get audio duration for progress estimation
        y, sr = librosa.load(audio_file)
        audio_duration = len(y) / sr
        print(f"[TASK {task_id}] Audio duration: {audio_duration:.2f} seconds")
        
        # Initialize madmom chord detection with high sensitivity settings
        # Configure for rapid chord change detection
        chroma_processor = DeepChromaProcessor(
            sample_rate=44100,  # Higher sample rate for better resolution
            hop_size=512,       # Smaller hop size for higher temporal resolution
            num_octaves=7,      # More octaves for better frequency coverage
            num_classes=12,     # Full chroma resolution
            fmin=27.5,          # Lower frequency bound (A0)
            fmax=3520.0,        # Higher frequency bound (A7)
            unique_filters=True  # Use unique filters for better accuracy
        )
        
        chord_processor = DeepChromaChordRecognitionProcessor(
            sample_rate=44100,  # Match chroma processor
            hop_size=512,       # Match chroma processor
            num_classes=25,     # Include more chord types
            unique_filters=True, # Use unique filters
            fps=50              # Higher frame rate for temporal resolution
        )
        
        # Start timing for progress estimation
        start_time = time.time()
        
        # Update progress: Starting chroma processing (40-45%)
        if task_id and task_id in tasks:
            tasks[task_id]['step'] = 'Analyzing chord pattern (chroma extraction)'
            tasks[task_id]['progress'] = 40
            print(f"[TASK {task_id}] Progress: 40% - Starting chroma extraction")
        
        # Process the audio file - chroma extraction
        chroma = chroma_processor(audio_file)
        
        # Update progress: Chroma extraction complete, starting chord recognition (45-50%)
        if task_id and task_id in tasks:
            tasks[task_id]['step'] = 'Analyzing chord pattern (chord recognition)'
            tasks[task_id]['progress'] = 45
            print(f"[TASK {task_id}] Progress: 45% - Chroma extraction complete, starting chord recognition")
        
        # Process chords - this is the most time-consuming part
        chords = chord_processor(chroma)
        
        # Calculate elapsed time and estimate remaining time
        elapsed_time = time.time() - start_time
        estimated_total_time = audio_duration * 0.1  # Rough estimate: 10% of audio duration for processing
        if elapsed_time < estimated_total_time:
            progress_ratio = elapsed_time / estimated_total_time
            estimated_progress = 45 + int(progress_ratio * 5)  # 45-50% range
            if task_id and task_id in tasks:
                tasks[task_id]['progress'] = estimated_progress
                print(f"[TASK {task_id}] Progress: {estimated_progress}% - Chord recognition in progress (estimated)")
        
        # Update progress: Raw chord detection complete (50-55%)
        if task_id and task_id in tasks:
            tasks[task_id]['step'] = 'Analyzing chord pattern (post-processing)'
            tasks[task_id]['progress'] = 50
            print(f"[TASK {task_id}] Progress: 50% - Raw chord detection complete, starting post-processing")
        
        # Debug: Print raw madmom output
        print(f"Raw madmom detected {len(chords)} chord segments:")
        for i, chord_data in enumerate(chords[:10]):  # Show first 10
            print(f"  {i}: {chord_data} (length: {len(chord_data)})")
        
        # Check the format of the first chord data
        if len(chords) > 0:
            print(f"First chord data format: {type(chords[0])}, length: {len(chords[0])}")
            print(f"First chord data: {chords[0]}")
        
        # Convert madmom output to our format
        chords_with_timing = []
        
        # Improved filtering parameters - More sensitive for better detection
        min_chord_duration = 0.2  # Even more reduced minimum duration to catch very short chord changes
        min_confidence = 0.2  # Even more reduced confidence threshold to catch more detections
        min_time_between_chords = 0.3  # Even more reduced minimum time between chord changes
        
        # Update progress: Starting chord filtering (55-60%)
        if task_id and task_id in tasks:
            tasks[task_id]['step'] = 'Analyzing chord pattern (filtering chords)'
            tasks[task_id]['progress'] = 55
            print(f"[TASK {task_id}] Progress: 55% - Starting chord filtering and validation")
        
        # First pass: collect all valid chord detections
        valid_chords = []
        filtered_out_count = 0
        for i, chord_data in enumerate(chords):
            try:
                # Handle different possible madmom output formats
                if len(chord_data) >= 4:
                    # Format: [start_time, end_time, chord_label, confidence]
                    start_time = float(chord_data[0])
                    end_time = float(chord_data[1])
                    chord_label = str(chord_data[2])
                    confidence = float(chord_data[3])
                elif len(chord_data) == 3:
                    # Format: [start_time, end_time, chord_label] (no confidence)
                    start_time = float(chord_data[0])
                    end_time = float(chord_data[1])
                    chord_label = str(chord_data[2])
                    confidence = 1.0  # Default confidence
                elif len(chord_data) == 2:
                    # Format: [time, chord_label]
                    start_time = float(chord_data[0])
                    end_time = float(chord_data[0]) + 1.0  # Assume 1 second duration
                    chord_label = str(chord_data[1])
                    confidence = 1.0  # Default confidence
                else:
                    print(f"Unexpected chord data format: {chord_data}")
                    continue
                
                # Skip 'N' (no chord) detections
                if chord_label == 'N':
                    filtered_out_count += 1
                    continue
                    
                # Filter by confidence threshold
                if confidence < min_confidence:
                    print(f"Filtered out low confidence chord: {chord_label} at {start_time:.2f}s (confidence: {confidence:.3f} < {min_confidence})")
                    filtered_out_count += 1
                    continue
                    
                # Skip very short chord detections
                chord_duration = end_time - start_time
                if chord_duration < min_chord_duration:
                    print(f"Filtered out short chord: {chord_label} at {start_time:.2f}s (duration: {chord_duration:.3f}s < {min_chord_duration}s)")
                    filtered_out_count += 1
                    continue
                    
                # Convert madmom chord labels to our format
                chord_name = convert_madmom_chord(chord_label)
                
                print(f"Processing chord: {chord_label} -> {chord_name} at {start_time:.2f}-{end_time:.2f}s (confidence: {confidence:.3f})")
                
                valid_chords.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'chord': chord_name,
                    'confidence': confidence,
                    'duration': chord_duration
                })
            except Exception as e:
                print(f"Error processing chord data {chord_data}: {e}")
                filtered_out_count += 1
                continue
        
        print(f"After filtering: {len(valid_chords)} valid chord segments (filtered out {filtered_out_count})")
        
        # Update progress: Starting chord smoothing (60-65%)
        if task_id and task_id in tasks:
            tasks[task_id]['step'] = 'Analyzing chord pattern (smoothing chords)'
            tasks[task_id]['progress'] = 60
            print(f"[TASK {task_id}] Progress: 60% - Starting chord smoothing and timing optimization")
        
        # Second pass: apply smoothing and timing logic
        if valid_chords:
            # Sort by start time
            valid_chords.sort(key=lambda x: x['start_time'])
            
            # Apply median filtering to reduce rapid switching - use smaller window for more sensitivity
            window_size = 2  # Reduced from 3 to 2 for less aggressive smoothing
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
                    print(f"Filtered out rapid chord change: {chord_info['chord']} at {chord_info['time']:.2f}s (too close to previous)")
            
            chords_with_timing = final_chords
        
        # Update progress: Chord detection complete (65%)
        if task_id and task_id in tasks:
            tasks[task_id]['step'] = 'Analyzing chord pattern (complete)'
            tasks[task_id]['progress'] = 65
            print(f"[TASK {task_id}] Progress: 65% - Chord detection and processing complete")
        
        print(f"Final result: {len(chords_with_timing)} chord changes")
        for chord_info in chords_with_timing:
            print(f"  {chord_info['chord']} at {chord_info['time']:.2f}s")
        
        # If madmom didn't detect enough chords, try to be more lenient
        if len(chords_with_timing) < 2:  # Reduced from 3 to 2
            print(f"Warning: Only detected {len(chords_with_timing)} chords, but continuing anyway")
            if len(chords_with_timing) == 0:
                raise RuntimeError(f"Madmom detected no chords. This could indicate an issue with the audio file or chord detection.")
        
        # Additional check: If no chords detected in first 10 seconds, try to force early detection
        early_chords = [c for c in chords_with_timing if c['time'] < 10.0]
        if len(early_chords) == 0 and len(chords_with_timing) > 0:
            print(f"Warning: No chords detected in first 10 seconds. Earliest chord at {chords_with_timing[0]['time']:.2f}s")
            # Try to add a chord at the beginning if we have any valid chords
            if len(valid_chords) > 0:
                # Find the earliest valid chord and add it at time 0
                earliest_valid = min(valid_chords, key=lambda x: x['start_time'])
                if earliest_valid['start_time'] > 5.0:  # If earliest is after 5s, add at 0
                    print(f"Adding early chord at 0.0s: {earliest_valid['chord']}")
                    chords_with_timing.insert(0, {
                        'time': 0.0,
                        'chord': earliest_valid['chord'],
                        'speech': chord_to_ipa_phonemes(earliest_valid['chord']),
                        'confidence': earliest_valid['confidence'],
                        'duration': earliest_valid['duration']
                    })
        
        return chords_with_timing
    except Exception as e:
        print(f"Madmom chord detection error: {e}")
        # Re-raise the exception to fail properly
        raise RuntimeError(f"Chord detection failed: {str(e)}")

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

def detect_chords_fallback(audio_file, chord_types=None):
    """Fallback chord detection using librosa if madmom fails"""
    if not lazy_import_audio_deps():
        print("Audio processing dependencies not available")
        return []
        
    try:
        y, sr = librosa.load(audio_file)
        
        # Extract chroma features with higher resolution
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=256)
        
        # Default chord type preferences if none provided
        if chord_types is None:
            chord_types = {
                'minor': True,
                'seventh': True,
                'minor_seventh': True,
                'major_seventh': True,
                'diminished': False,
                'augmented': False,
                'suspended': False,
                'power': False,
                'add_nine': False,
                'sixth': False,
                'minor_sixth': False
            }
        
        # Simple chord detection based on chroma peaks
        chord_templates = {
            # Major chords (natural notes)
            'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            'D': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            'E': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            'F': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            'G': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            'A': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            'B': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            
            # Major chords (sharps and flats)
            'C#': [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # C#/Db
            'Db': [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],   # Same as C#
            'D#': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],   # D#/Eb
            'Eb': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],   # Same as D#
            'F#': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],   # F#/Gb
            'Gb': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],   # Same as F#
            'G#': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],   # G#/Ab
            'Ab': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],   # Same as G#
            'A#': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],   # A#/Bb
            'Bb': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],   # Same as A#
            
            # Minor chords (natural notes)
            'Cm': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            'Dm': [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            'Em': [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            'Fm': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            'Gm': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            'Am': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            'Bm': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            
            # Minor chords (sharps and flats)
            'C#m': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # C#m/Dbm
            'Dbm': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # Same as C#m
            'D#m': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # D#m/Ebm
            'Ebm': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # Same as D#m
            'F#m': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],  # F#m/Gbm
            'Gbm': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],  # Same as F#m
            'G#m': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # G#m/Abm
            'Abm': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # Same as G#m
            'A#m': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # A#m/Bbm
            'Bbm': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # Same as A#m
            
            # Seventh chords (dominant) - natural notes
            'C7': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            'D7': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            'E7': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            'F7': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            'G7': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            'A7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            'B7': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            
            # Seventh chords (dominant) - sharps and flats
            'C#7': [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],  # C#7/Db7
            'Db7': [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],   # Same as C#7
            'D#7': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],   # D#7/Eb7
            'Eb7': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],   # Same as D#7
            'F#7': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],   # F#7/Gb7
            'Gb7': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],   # Same as F#7
            'G#7': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],   # G#7/Ab7
            'Ab7': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],   # Same as G#7
            'A#7': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],   # A#7/Bb7
            'Bb7': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],   # Same as A#7
            
            # Minor seventh chords - natural notes
            'Cm7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            'Dm7': [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            'Em7': [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            'Fm7': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            'Gm7': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            'Am7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            'Bm7': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            
            # Minor seventh chords - sharps and flats
            'C#m7': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # C#m7/Dbm7
            'Dbm7': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # Same as C#m7
            'D#m7': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # D#m7/Ebm7
            'Ebm7': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # Same as D#m7
            'F#m7': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],  # F#m7/Gbm7
            'Gbm7': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],  # Same as F#m7
            'G#m7': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # G#m7/Abm7
            'Abm7': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # Same as G#m7
            'A#m7': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # A#m7/Bbm7
            'Bbm7': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # Same as A#m7
            
            # Major seventh chords - natural notes
            'Cmaj7': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            'Dmaj7': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            'Emaj7': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            'Fmaj7': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            'Gmaj7': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            'Amaj7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            'Bmaj7': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            
            # Major seventh chords - sharps and flats
            'C#maj7': [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # C#maj7/Dbmaj7
            'Dbmaj7': [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],   # Same as C#maj7
            'D#maj7': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],   # D#maj7/Ebmaj7
            'Ebmaj7': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],   # Same as D#maj7
            'F#maj7': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],   # F#maj7/Gbmaj7
            'Gbmaj7': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],   # Same as F#maj7
            'G#maj7': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],   # G#maj7/Abmaj7
            'Abmaj7': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],   # Same as G#maj7
            'A#maj7': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],   # A#maj7/Bbmaj7
            'Bbmaj7': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],   # Same as A#maj7
            
            # Diminished chords - natural notes
            'Cdim': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            'Ddim': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            'Edim': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            'Fdim': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            'Gdim': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            'Adim': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            'Bdim': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            
            # Diminished chords - sharps and flats
            'C#dim': [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # C#dim/Dbdim
            'Dbdim': [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # Same as C#dim
            'D#dim': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # D#dim/Ebdim
            'Ebdim': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # Same as D#dim
            'F#dim': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],  # F#dim/Gbdim
            'Gbdim': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],  # Same as F#dim
            'G#dim': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # G#dim/Abdim
            'Abdim': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # Same as G#dim
            'A#dim': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],  # A#dim/Bbdim
            'Bbdim': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],  # Same as A#dim
            
            # Augmented chords - natural notes
            'Caug': [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            'Daug': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            'Eaug': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            'Faug': [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            'Gaug': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            'Aaug': [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            'Baug': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            
            # Augmented chords - sharps and flats
            'C#aug': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],  # C#aug/Dbaug
            'Dbaug': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],  # Same as C#aug
            'D#aug': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # D#aug/Ebaug
            'Ebaug': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # Same as D#aug
            'F#aug': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # F#aug/Gbaug
            'Gbaug': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Same as F#aug
            'G#aug': [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # G#aug/Abaug
            'Abaug': [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # Same as G#aug
            'A#aug': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # A#aug/Bbaug
            'Bbaug': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Same as A#aug
            
            # Suspended chords (sus2) - natural notes
            'Csus2': [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'Dsus2': [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            'Esus2': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            'Fsus2': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            'Gsus2': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            'Asus2': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            'Bsus2': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            
            # Suspended chords (sus2) - sharps and flats
            'C#sus2': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # C#sus2/Dbsus2
            'Dbsus2': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # Same as C#sus2
            'D#sus2': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # D#sus2/Ebsus2
            'Ebsus2': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # Same as D#sus2
            'F#sus2': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # F#sus2/Gbsus2
            'Gbsus2': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # Same as F#sus2
            'G#sus2': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # G#sus2/Absus2
            'Absus2': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # Same as G#sus2
            'A#sus2': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # A#sus2/Bbsus2
            'Bbsus2': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # Same as A#sus2
            
            # Suspended chords (sus4) - natural notes
            'Csus4': [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            'Dsus4': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            'Esus4': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            'Fsus4': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            'Gsus4': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            'Asus4': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            'Bsus4': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            
            # Suspended chords (sus4) - sharps and flats
            'C#sus4': [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],  # C#sus4/Dbsus4
            'Dbsus4': [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],  # Same as C#sus4
            'D#sus4': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # D#sus4/Ebsus4
            'Ebsus4': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # Same as D#sus4
            'F#sus4': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # F#sus4/Gbsus4
            'Gbsus4': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # Same as F#sus4
            'G#sus4': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # G#sus4/Absus4
            'Absus4': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # Same as G#sus4
            'A#sus4': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # A#sus4/Bbsus4
            'Bbsus4': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # Same as A#sus4
            
            # Power chords (5th) - natural notes
            'C5': [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'D5': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            'E5': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            'F5': [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            'G5': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'A5': [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            'B5': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            
            # Power chords (5th) - sharps and flats
            'C#5': [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # C#5/Db5
            'Db5': [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Same as C#5
            'D#5': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # D#5/Eb5
            'Eb5': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # Same as D#5
            'F#5': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # F#5/Gb5
            'Gb5': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Same as F#5
            'G#5': [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # G#5/Ab5
            'Ab5': [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Same as G#5
            'A#5': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # A#5/Bb5
            'Bb5': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Same as A#5
            
            # Add9 chords - natural notes
            'Cadd9': [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            'Dadd9': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            'Eadd9': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            'Fadd9': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            'Gadd9': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            'Aadd9': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            'Badd9': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            
            # Add9 chords - sharps and flats
            'C#add9': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # C#add9/Dbadd9
            'Dbadd9': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # Same as C#add9
            'D#add9': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # D#add9/Ebadd9
            'Ebadd9': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # Same as D#add9
            'F#add9': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # F#add9/Gbadd9
            'Gbadd9': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # Same as F#add9
            'G#add9': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # G#add9/Abadd9
            'Abadd9': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # Same as G#add9
            'A#add9': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # A#add9/Bbadd9
            'Bbadd9': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # Same as A#add9
            
            # 6th chords - natural notes
            'C6': [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
            'D6': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            'E6': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            'F6': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            'G6': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            'A6': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            'B6': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            
            # 6th chords - sharps and flats
            'C#6': [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],  # C#6/Db6
            'Db6': [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],  # Same as C#6
            'D#6': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # D#6/Eb6
            'Eb6': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # Same as D#6
            'F#6': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # F#6/Gb6
            'Gb6': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # Same as F#6
            'G#6': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # G#6/Ab6
            'Ab6': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # Same as G#6
            'A#6': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # A#6/Bb6
            'Bb6': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # Same as A#6
            
            # Minor 6th chords - natural notes
            'Cm6': [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
            'Dm6': [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            'Em6': [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            'Fm6': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            'Gm6': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            'Am6': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            'Bm6': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            
            # Minor 6th chords - sharps and flats
            'C#m6': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],  # C#m6/Dbm6
            'Dbm6': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],  # Same as C#m6
            'D#m6': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # D#m6/Ebm6
            'Ebm6': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # Same as D#m6
            'F#m6': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],  # F#m6/Gbm6
            'Gbm6': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],  # Same as F#m6
            'G#m6': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # G#m6/Abm6
            'Abm6': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # Same as G#m6
            'A#m6': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # A#m6/Bbm6
            'Bbm6': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]   # Same as A#m6
        }
        
        # Filter chord templates based on user preferences
        filtered_templates = {}
        
        # Always include major chords (natural notes and sharps/flats)
        for chord in ['C', 'D', 'E', 'F', 'G', 'A', 'B', 
                     'C#', 'Db', 'D#', 'Eb', 'F#', 'Gb', 'G#', 'Ab', 'A#', 'Bb']:
            if chord in chord_templates:
                filtered_templates[chord] = chord_templates[chord]
        
        # Include other chord types based on user preferences
        for chord, template in chord_templates.items():
            if chord in filtered_templates:  # Skip major chords (already included)
                continue
                
            # Check if this chord type should be included
            include_chord = False
            
            if chord.endswith('m') and not chord.endswith('m7') and not chord.endswith('m6') and not chord.endswith('maj7'):
                include_chord = chord_types.get('minor', True)
            elif chord.endswith('7') and not chord.endswith('m7') and not chord.endswith('maj7'):
                include_chord = chord_types.get('seventh', True)
            elif chord.endswith('m7'):
                include_chord = chord_types.get('minor_seventh', True)
            elif chord.endswith('maj7'):
                include_chord = chord_types.get('major_seventh', True)
            elif chord.endswith('dim'):
                include_chord = chord_types.get('diminished', False)
            elif chord.endswith('aug'):
                include_chord = chord_types.get('augmented', False)
            elif chord.startswith(('C', 'D', 'E', 'F', 'G', 'A', 'B')) and ('sus' in chord):
                include_chord = chord_types.get('suspended', False)
            elif chord.endswith('5'):
                include_chord = chord_types.get('power', False)
            elif chord.endswith('add9'):
                include_chord = chord_types.get('add_nine', False)
            elif chord.endswith('6') and not chord.endswith('m6'):
                include_chord = chord_types.get('sixth', False)
            elif chord.endswith('m6'):
                include_chord = chord_types.get('minor_sixth', False)
            
            if include_chord:
                filtered_templates[chord] = template
        
        # Use filtered templates
        chord_templates = filtered_templates
        
        # Detect onsets for more precise timing
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=256, units='frames')
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=256)
        
        # Also get beat times as fallback
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=256)
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=256)
        
        # Combine onset and beat times, prioritizing onsets
        all_times = sorted(list(set(onset_times.tolist() + beat_times.tolist())))
        
        # Filter times to avoid too frequent chord changes (minimum 1.0 seconds apart for full speech)
        filtered_times = []
        last_time = -1
        for time in all_times:
            if time - last_time >= 1.0:  # Minimum 1.0 seconds between chord changes for full speech
                filtered_times.append(time)
                last_time = time
        
        # Detect chord for each filtered time point
        chords_with_timing = []
        
        for time in filtered_times:
            frame_idx = int(time * sr / 256)
            if frame_idx < chroma.shape[1]:
                frame_chroma = chroma[:, frame_idx]
                
                # Find best matching chord
                best_chord = 'C'
                best_score = 0
                for chord, template in chord_templates.items():
                    score = np.dot(frame_chroma, template)
                    if score > best_score:
                        best_score = score
                        best_chord = chord
                
                # Add chord announcement for every detected time point
                chords_with_timing.append({
                    'time': float(time),
                    'chord': best_chord,
                    'speech': chord_to_ipa_phonemes(best_chord)
                })
        
        return chords_with_timing
    except Exception as e:
        print(f"Chord detection error: {e}")
        return []

def separate_vocals_demucs(audio_path, output_dir, task_id=None):
    """Separate vocals using Demucs (high-quality vocal separation).
    Outputs are saved as 'vocal_track.wav' (vocals only) and 'instrumental_track.wav' (instrumental only) in the output directory."""
    try:
        import subprocess
        import tempfile
        import glob
        import shutil
        import signal
        
        log_debug(task_id, f"Starting Demucs vocal separation for {audio_path}")
        log_debug(task_id, f"Output directory: {output_dir}")
        
        # Use Demucs command line interface with timeout
        cmd = [
            'demucs',
            '--two-stems=vocals',  # Separate vocals from the rest
            '--out', output_dir,
            '--mp3',  # Use MP3 output for faster processing
            '--mp3-bitrate', '128',  # Lower bitrate for speed
            audio_path
        ]
        log_debug(task_id, f"Running Demucs command: {' '.join(cmd)}")
        print(f"Running Demucs command: {' '.join(cmd)}")
        
        # Update task status if task_id is provided
        if task_id and task_id in tasks:
            tasks[task_id]['step'] = 'Splitting vocal & instrumental'
            tasks[task_id]['demucs_percentage'] = 0  # Initialize demucs percentage
            tasks[task_id]['progress'] = 10  # Start at 10% for demucs step
            print(f"[TASK {task_id}] Starting demucs with progress: 10%")
        
        # Run with timeout (15 minutes max) and show real-time output
        try:
            # Use Popen to get real-time output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                     text=True, bufsize=1, universal_newlines=True)
            
            stdout_lines = []
            stderr_lines = []
            
            # Read output in real-time
            while True:
                stdout_line = process.stdout.readline()
                stderr_line = process.stderr.readline()
                
                if stdout_line:
                    stdout_lines.append(stdout_line.strip())
                    log_debug(task_id, f"Demucs stdout: {stdout_line.strip()}")
                    print(f"[TASK {task_id}] Demucs stdout: {stdout_line.strip()}")
                    
                    # Update task status with progress if we detect progress info
                    if task_id and task_id in tasks:
                        # Try multiple patterns for progress detection
                        progress_found = False
                        
                        # Pattern 1: "25%|██▎ | 5.85/187.2"
                        match = re.search(r'(\d+)%', stdout_line)
                        if match:
                            demucs_percentage = int(match.group(1))
                            progress_found = True
                        
                        # Pattern 2: "Processing: 25%" or "Progress: 25%"
                        if not progress_found:
                            match = re.search(r'(?:Processing|Progress):\s*(\d+)%', stdout_line, re.IGNORECASE)
                            if match:
                                demucs_percentage = int(match.group(1))
                                progress_found = True
                        
                        # Pattern 3: "25/100" or "5.85/187.2" (fraction format)
                        if not progress_found:
                            match = re.search(r'(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)', stdout_line)
                            if match:
                                current = float(match.group(1))
                                total = float(match.group(2))
                                if total > 0:
                                    demucs_percentage = int((current / total) * 100)
                                    progress_found = True
                        
                        if progress_found:
                            # Store demucs percentage in task for frontend
                            tasks[task_id]['demucs_percentage'] = demucs_percentage
                            # Map demucs progress (0-100) to overall progress (10-25) with proper linear scaling
                            # Formula: overall = 10 + (demucs_percentage * 15 / 100)
                            # This maps: 0%->10%, 25%->13.75%, 50%->17.5%, 75%->21.25%, 100%->25%
                            overall_progress = 10 + int((demucs_percentage * 15) / 100)
                            
                            # Only update if new progress is higher than current
                            current_progress = tasks[task_id].get('progress', 0)
                            if overall_progress > current_progress:
                                tasks[task_id]['progress'] = overall_progress
                                tasks[task_id]['step'] = 'Splitting vocal & instrumental'
                                print(f"[TASK {task_id}] Demucs progress: {demucs_percentage}% -> Overall: {overall_progress}%")
                            else:
                                # Even if no progress found, update step to show we're still working
                                tasks[task_id]['step'] = 'Splitting vocal & instrumental'
                                print(f"[TASK {task_id}] Demucs output (no progress): {stdout_line.strip()}")
                    
                if stderr_line:
                    stderr_lines.append(stderr_line.strip())
                    log_debug(task_id, f"Demucs stderr: {stderr_line.strip()}")
                    print(f"[TASK {task_id}] Demucs stderr: {stderr_line.strip()}")
                    
                    # Also check stderr for progress information
                    if task_id and task_id in tasks:
                        # Try multiple patterns for progress detection in stderr
                        progress_found = False
                        
                        # Pattern 1: "25%|██▎ | 5.85/187.2"
                        match = re.search(r'(\d+)%', stderr_line)
                        if match:
                            demucs_percentage = int(match.group(1))
                            progress_found = True
                        
                        # Pattern 2: "Processing: 25%" or "Progress: 25%"
                        if not progress_found:
                            match = re.search(r'(?:Processing|Progress):\s*(\d+)%', stderr_line, re.IGNORECASE)
                            if match:
                                demucs_percentage = int(match.group(1))
                                progress_found = True
                        
                        # Pattern 3: "25/100" or "5.85/187.2" (fraction format)
                        if not progress_found:
                            match = re.search(r'(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)', stderr_line)
                            if match:
                                current = float(match.group(1))
                                total = float(match.group(2))
                                if total > 0:
                                    demucs_percentage = int((current / total) * 100)
                                    progress_found = True
                        
                        if progress_found:
                            # Store demucs percentage in task for frontend
                            tasks[task_id]['demucs_percentage'] = demucs_percentage
                            # Map demucs progress (0-100) to overall progress (10-25) with proper linear scaling
                            # Formula: overall = 10 + (demucs_percentage * 15 / 100)
                            # This maps: 0%->10%, 25%->13.75%, 50%->17.5%, 75%->21.25%, 100%->25%
                            overall_progress = 10 + int((demucs_percentage * 15) / 100)
                            
                            # Only update if new progress is higher than current
                            current_progress = tasks[task_id].get('progress', 0)
                            if overall_progress > current_progress:
                                tasks[task_id]['progress'] = overall_progress
                                tasks[task_id]['step'] = 'Splitting vocal & instrumental'
                                print(f"[TASK {task_id}] Demucs progress (stderr): {demucs_percentage}% -> Overall: {overall_progress}%")
                    
                    # Check if process has finished
                    if process.poll() is not None:
                        break
            
            # Wait for process to complete
            result = process.wait(timeout=900)
            
            if result != 0:
                print(f"Demucs command failed with return code {result}")
                return None, None
                
        except subprocess.TimeoutExpired:
            log_debug(task_id, "Demucs command timed out after 15 minutes")
            print("Demucs command timed out after 15 minutes")
            if task_id and task_id in tasks:
                tasks[task_id]['step'] = 'Demucs timed out after 15 minutes'
            return None, None
        except Exception as e:
            log_debug(task_id, f"Demucs subprocess error: {e}")
            print(f"Demucs subprocess error: {e}")
            if task_id and task_id in tasks:
                tasks[task_id]['step'] = f'Demucs error: {str(e)}'
            return None, None
            
        demucs_output = os.path.join(output_dir, 'htdemucs')
        log_debug(task_id, f"Looking for demucs output in: {demucs_output}")
        print(f"Looking for demucs output in: {demucs_output}")
        
        if not os.path.exists(demucs_output):
            log_debug(task_id, f"Demucs output directory not found: {demucs_output}")
            print(f"Demucs output directory not found: {demucs_output}")
            # List contents of output_dir to see what was created
            if os.path.exists(output_dir):
                contents = os.listdir(output_dir)
                log_debug(task_id, f"Contents of {output_dir}: {contents}")
                print(f"Contents of {output_dir}: {contents}")
            return None, None
            
        # List contents of demucs_output to see what's there
        demucs_contents = os.listdir(demucs_output)
        log_debug(task_id, f"Contents of demucs output directory: {demucs_contents}")
        print(f"Contents of demucs output directory: {demucs_contents}")
        
        # Look for both .wav and .mp3 files
        vocals_files = glob.glob(os.path.join(demucs_output, '*', 'vocals.*'))
        no_vocals_files = glob.glob(os.path.join(demucs_output, '*', 'no_vocals.*'))
        
        log_debug(task_id, f"Found vocals files: {vocals_files}")
        log_debug(task_id, f"Found no_vocals files: {no_vocals_files}")
        print(f"Found vocals files: {vocals_files}")
        print(f"Found no_vocals files: {no_vocals_files}")
        
        if not vocals_files or not no_vocals_files:
            log_debug(task_id, f"Demucs output files not found. Vocals: {vocals_files}, No vocals: {no_vocals_files}")
            print(f"Demucs output files not found. Vocals: {vocals_files}, No vocals: {no_vocals_files}")
            # Try to find any files in the demucs output
            all_files = []
            for root, dirs, files in os.walk(demucs_output):
                for file in files:
                    all_files.append(os.path.join(root, file))
            log_debug(task_id, f"All files in demucs output: {all_files}")
            print(f"All files in demucs output: {all_files}")
            return None, None
            
        # Copy to standardized names
        vocal_track_path = os.path.join(output_dir, 'vocal_track.wav')
        instrumental_track_path = os.path.join(output_dir, 'instrumental_track.wav')
        
        # Convert to WAV if needed
        vocals_source = vocals_files[0]
        no_vocals_source = no_vocals_files[0]
        
        if vocals_source.endswith('.mp3'):
            # Convert MP3 to WAV
            audio = AudioSegment.from_mp3(vocals_source)
            audio.export(vocal_track_path, format='wav')
        else:
            shutil.copy2(vocals_source, vocal_track_path)
            
        if no_vocals_source.endswith('.mp3'):
            # Convert MP3 to WAV
            audio = AudioSegment.from_mp3(no_vocals_source)
            audio.export(instrumental_track_path, format='wav')
        else:
            shutil.copy2(no_vocals_source, instrumental_track_path)
            
        log_debug(task_id, f"Vocal separation completed: {vocal_track_path}, {instrumental_track_path}")
        print(f"Vocal separation completed: {vocal_track_path}, {instrumental_track_path}")
        return vocal_track_path, instrumental_track_path
        
    except Exception as e:
        print(f"Demucs vocal separation error: {e}")
        return None, None



def process_audio_task(task_id, file_path):
    """Background task to process audio file"""
    log_debug(task_id, "Starting audio processing task")
    
    if not lazy_import_audio_deps():
        error_msg = "Audio processing dependencies not available. Cannot process audio."
        log_debug(task_id, f"ERROR: {error_msg}")
        print(f"ERROR: {error_msg}")
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = error_msg
        tasks[task_id]['step'] = f'Error: {error_msg}'
        return
        
    try:
        import time
        start_time = time.time()
        
        print(f"\n=== [TASK {task_id}] STARTING PROCESSING ===")
        print(f"[TASK {task_id}] Input file: {file_path}")
        print(f"[TASK {task_id}] File size: {os.path.getsize(file_path)} bytes")
        
        tasks[task_id]['status'] = 'processing'
        task_dir = os.path.join(UPLOAD_FOLDER, task_id)
        
        # Step 1: Convert to wav if needed
        print(f"\n=== [TASK {task_id}] STEP 1: PREPARING AUDIO FILE ===")
        tasks[task_id]['step'] = 'Preparing audio file'
        tasks[task_id]['progress'] = 5
        print(f"[TASK {task_id}] Progress: 5% - Preparing audio file")
        
        step1_start = time.time()
        print(f"[TASK {task_id}] Loading audio with pydub...")
        audio = AudioSegment.from_file(file_path)
        print(f"[TASK {task_id}] Audio loaded successfully. Duration: {len(audio)/1000:.2f} seconds")
        
        wav_path = os.path.join(task_dir, 'input.wav')
        print(f"[TASK {task_id}] Exporting to WAV: {wav_path}")
        audio.export(wav_path, format='wav')
        step1_time = time.time() - step1_start
        print(f"[TASK {task_id}] WAV export completed in {step1_time:.2f} seconds")
        print(f"[TASK {task_id}] WAV file size: {os.path.getsize(wav_path)} bytes")
        
        # Step 2: Vocal separation using Demucs only
        print(f"Step 2: Starting vocal separation for task {task_id}")
        tasks[task_id]['step'] = 'Splitting vocal & instrumental'
        tasks[task_id]['progress'] = 10
        print(f"[TASK {task_id}] Progress: 10% - Starting vocal separation")
        vocals_path, instrumental_path = separate_vocals_demucs(wav_path, task_dir, task_id)
        log_debug(task_id, f"Demucs returned: vocals={vocals_path}, instrumental={instrumental_path}")
        if vocals_path is None or instrumental_path is None:
            error_msg = "Demucs vocal separation failed. Cannot proceed without proper vocal separation."
            log_debug(task_id, f"ERROR: {error_msg}")
            print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)
        log_debug(task_id, f"Vocal separation completed: {vocals_path}, {instrumental_path}")
        print(f"Vocal separation completed: {vocals_path}, {instrumental_path}")
        
        # Step 3: Extract voice sample from vocals for voice cloning
        print(f"Step 3: Extracting voice sample for task {task_id}")
        tasks[task_id]['step'] = 'Extracting voice sample'
        tasks[task_id]['progress'] = 30
        print(f"[TASK {task_id}] Progress: 30% - Extracting voice sample")
        voice_sample, voice_sr = extract_voice_sample(vocals_path)
        voice_sample_path = None
        if voice_sample is not None:
            voice_sample_path = os.path.join(task_dir, 'voice_sample.wav')
            write_wav(voice_sample_path, voice_sr, voice_sample)
            print(f"Voice sample extracted: {voice_sample_path}")
        else:
            print("WARNING: Voice sample extraction failed")
        
        # Step 4: Chord detection (using instrumental track)
        print(f"\n=== [TASK {task_id}] STEP 4: CHORD DETECTION ===")
        tasks[task_id]['step'] = 'Analyzing chord pattern'
        tasks[task_id]['progress'] = 40
        print(f"[TASK {task_id}] Progress: 40% - Analyzing chord pattern")
        
        chord_start = time.time()
        print(f"[TASK {task_id}] Starting chord detection with madmom...")
        print(f"[TASK {task_id}] Instrumental file: {instrumental_path}")
        print(f"[TASK {task_id}] Instrumental file size: {os.path.getsize(instrumental_path)} bytes")
        
        try:
            # Use madmom's default detection with progress tracking
            chords = detect_chords(instrumental_path, task_id=task_id)
            chord_time = time.time() - chord_start
            print(f"[TASK {task_id}] Chord detection completed in {chord_time:.2f} seconds")
            print(f"[TASK {task_id}] Detected {len(chords)} chords")
            tasks[task_id]['progress'] = 65
            print(f"[TASK {task_id}] Progress: 65% - Chord detection completed")
        except Exception as chord_error:
            print(f"[TASK {task_id}] ERROR in chord detection: {chord_error}")
            raise RuntimeError(f"Chord detection failed: {str(chord_error)}")
        
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
    return jsonify({
        'status': 'healthy',
        'version': VERSION,
        'name': 'ChordiSpeak'
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
