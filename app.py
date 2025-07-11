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
from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
from werkzeug.utils import secure_filename
import librosa
import numpy as np
from pydub import AudioSegment
import subprocess
import tempfile
import shutil
from threading import Thread
import time
from scipy.io.wavfile import write as write_wav

def get_version():
    """Read version from VERSION file"""
    try:
        with open('VERSION', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return "1.0.0"  # fallback version

VERSION = get_version()

# Try to import TTS, error if not available
try:
    import torch
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Coqui TTS not available. Please install Coqui TTS (XTTS v2) to use this app.")

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Task storage (in production, use Redis or database)
tasks = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def chord_to_speech(chord):
    """Convert chord notation to phonetic speech"""
    chord_map = {
        'C': 'SEE', 'C#': 'SEE SHARP', 'Db': 'DEE FLAT',
        'D': 'DEE', 'D#': 'DEE SHARP', 'Eb': 'EE FLAT',
        'E': 'EE', 'F': 'EFF', 'F#': 'EFF SHARP', 'Gb': 'GEE FLAT',
        'G': 'GEE', 'G#': 'GEE SHARP', 'Ab': 'AY FLAT',
        'A': 'AYE', 'A#': 'AYE SHARP', 'Bb': 'BEE FLAT',
        'B': 'BEE'
    }
    
    # Handle special chord types first
    if chord.endswith('dim'):
        base_chord = chord[:-3]  # Remove 'dim'
        return chord_map.get(base_chord, base_chord) + ' DIMINISHED'
    elif chord.endswith('aug'):
        base_chord = chord[:-3]  # Remove 'aug'
        return chord_map.get(base_chord, base_chord) + ' AUGMENTED'
    elif chord.endswith('sus2'):
        base_chord = chord[:-4]  # Remove 'sus2'
        return chord_map.get(base_chord, base_chord) + ' SUSPENDED TWO'
    elif chord.endswith('sus4'):
        base_chord = chord[:-4]  # Remove 'sus4'
        return chord_map.get(base_chord, base_chord) + ' SUSPENDED FOUR'
    elif chord.endswith('sus'):
        base_chord = chord[:-3]  # Remove 'sus'
        return chord_map.get(base_chord, base_chord) + ' SUSPENDED'
    elif chord.endswith('maj7'):
        base_chord = chord[:-4]  # Remove 'maj7'
        return chord_map.get(base_chord, base_chord) + ' MAJOR SEVENTH'
    elif chord.endswith('m7'):
        base_chord = chord[:-2]  # Remove 'm7'
        return chord_map.get(base_chord, base_chord) + ' MINOR SEVENTH'
    elif chord.endswith('7'):
        base_chord = chord[:-1]  # Remove '7'
        return chord_map.get(base_chord, base_chord) + ' SEVENTH'
    elif chord.endswith('m6'):
        base_chord = chord[:-2]  # Remove 'm6'
        return chord_map.get(base_chord, base_chord) + ' MINOR SIXTH'
    elif chord.endswith('6'):
        base_chord = chord[:-1]  # Remove '6'
        return chord_map.get(base_chord, base_chord) + ' SIXTH'
    elif chord.endswith('add9'):
        base_chord = chord[:-4]  # Remove 'add9'
        return chord_map.get(base_chord, base_chord) + ' ADD NINE'
    elif chord.endswith('5'):
        base_chord = chord[:-1]  # Remove '5'
        return chord_map.get(base_chord, base_chord) + ' POWER'
    elif chord.endswith('m'):
        base_chord = chord[:-1]  # Remove 'm'
        return chord_map.get(base_chord, base_chord) + ' MINOR'
    else:
        # Major chord (no suffix)
        return chord_map.get(chord, chord)

def extract_voice_sample(vocals_path, sample_duration=None):
    """Extract voice sample from separated vocals for voice cloning"""
    try:
        # Use the full vocal track for better voice cloning
        y, sr = librosa.load(vocals_path)
        
        # Return the entire vocal track
        return y, sr
        
    except Exception as e:
        print(f"Voice sample extraction error: {e}")
        return None, None

def synthesize_chord_speech_coqui(text, voice_sample_path, output_path):
    """Generate speech using Coqui XTTS v2 voice cloning"""
    if not TTS_AVAILABLE:
        raise RuntimeError("Coqui TTS not available. Please install Coqui TTS (XTTS v2).")
    try:
        # Fix for PyTorch 2.6+ weights_only issue - monkey patch torch.load
        import torch
        original_torch_load = torch.load
        
        def patched_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        torch.load = patched_torch_load
        
        # Initialize TTS with XTTS v2 model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        
        # Generate speech
        tts.tts_to_file(
            text=text,
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
    if not TTS_AVAILABLE:
        raise RuntimeError("Coqui TTS not available. Please install Coqui TTS (XTTS v2).")
    if not voice_sample_path or not os.path.exists(voice_sample_path):
        raise RuntimeError("Voice sample for cloning not found. Cannot synthesize without a reference voice.")
    return synthesize_chord_speech_coqui(text, voice_sample_path, output_path)

def detect_chords(audio_file, chord_types=None):
    """Detect chords from audio file using librosa with improved timing"""
    try:
        y, sr = librosa.load(audio_file)
        
        # Extract chroma features with higher resolution
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=256)  # Smaller hop for better timing
        
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
        
        # Filter times to avoid too frequent chord changes (minimum 0.5 seconds apart)
        filtered_times = []
        last_time = -1
        for time in all_times:
            if time - last_time >= 0.5:  # Minimum 0.5 seconds between chord changes
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
                    'speech': chord_to_speech(best_chord)
                })
        
        return chords_with_timing
    except Exception as e:
        print(f"Chord detection error: {e}")
        return []

def separate_vocals_demucs(audio_path, output_dir):
    """Separate vocals using Demucs (high-quality vocal separation).
    Outputs are saved as 'vocal_track.wav' (vocals only) and 'instrumental_track.wav' (instrumental only) in the output directory."""
    try:
        import subprocess
        import tempfile
        import glob
        import shutil
        # Use Demucs command line interface
        cmd = [
            'demucs',
            '--two-stems=vocals',  # Separate vocals from the rest
            '--out', output_dir,
            audio_path
        ]
        print(f"Running Demucs command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Demucs command failed: {result.stderr}")
            return None, None
        demucs_output = os.path.join(output_dir, 'htdemucs')
        if not os.path.exists(demucs_output):
            print(f"Demucs output directory not found: {demucs_output}")
            return None, None
        vocals_files = glob.glob(os.path.join(demucs_output, '*', 'vocals.wav'))
        no_vocals_files = glob.glob(os.path.join(demucs_output, '*', 'no_vocals.wav'))
        if not vocals_files or not no_vocals_files:
            print("Demucs output files not found")
            return None, None
        # Copy to standardized names
        vocal_track_path = os.path.join(output_dir, 'vocal_track.wav')
        instrumental_track_path = os.path.join(output_dir, 'instrumental_track.wav')
        shutil.copy2(vocals_files[0], vocal_track_path)
        shutil.copy2(no_vocals_files[0], instrumental_track_path)
        return vocal_track_path, instrumental_track_path
    except Exception as e:
        print(f"Demucs vocal separation error: {e}")
        return None, None



def process_audio_task(task_id, file_path):
    """Background task to process audio file"""
    try:
        tasks[task_id]['status'] = 'processing'
        task_dir = os.path.join(UPLOAD_FOLDER, task_id)
        
        # Step 1: Convert to wav if needed
        tasks[task_id]['step'] = 'Preparing audio file'
        
        audio = AudioSegment.from_file(file_path)
        wav_path = os.path.join(task_dir, 'input.wav')
        audio.export(wav_path, format='wav')
        
        # Step 2: Vocal separation using Demucs only
        tasks[task_id]['step'] = 'Splitting vocal & instrumental'
        
        # Use Demucs for vocal separation - no fallbacks
        vocals_path, instrumental_path = separate_vocals_demucs(wav_path, task_dir)
        
        if vocals_path is None or instrumental_path is None:
            raise RuntimeError("Demucs vocal separation failed. Cannot proceed without proper vocal separation.")
        
        # Step 3: Extract voice sample from vocals for voice cloning
        tasks[task_id]['step'] = 'Extracting voice sample'
        
        voice_sample, voice_sr = extract_voice_sample(vocals_path)
        voice_sample_path = None
        
        if voice_sample is not None:
            voice_sample_path = os.path.join(task_dir, 'voice_sample.wav')
            write_wav(voice_sample_path, voice_sr, voice_sample)
        
        # Step 4: Chord detection (using instrumental track)
        tasks[task_id]['step'] = 'Analyzing chord pattern'
        
        # Get chord type preferences from task
        chord_types = tasks[task_id].get('chord_types', None)
        chords = detect_chords(instrumental_path, chord_types)
        
        # Save chord data
        chords_file = os.path.join(task_dir, 'chords.json')
        with open(chords_file, 'w') as f:
            json.dump(chords, f)
        
        # Step 5: Voice synthesis using voice cloning
        tasks[task_id]['step'] = 'Synthesizing spoken chord overlay'
        
        # Generate TTS for each unique chord
        unique_chords = list(set(chord_data['speech'] for chord_data in chords))
        tts_cache = {}
        
        for chord_speech in unique_chords:
            tts_output_path = os.path.join(task_dir, f'tts_{chord_speech.replace(" ", "_").replace("#", "sharp")}.wav')
            if not synthesize_chord_speech(chord_speech, voice_sample_path, tts_output_path):
                raise RuntimeError(f"TTS synthesis failed for chord: {chord_speech}")
            # Load the generated TTS audio
            if os.path.exists(tts_output_path):
                tts_cache[chord_speech] = AudioSegment.from_wav(tts_output_path)
            else:
                raise RuntimeError(f"TTS output file not created for chord: {chord_speech}")
        
        # Create chord audio track with proper timing
        chord_audio_segments = []
        
        for i, chord_data in enumerate(chords):
            # Calculate precise timing
            if i == 0:
                silence_duration = chord_data['time'] * 1000  # Convert to ms
            else:
                silence_duration = (chord_data['time'] - chords[i-1]['time']) * 1000
            
            # Ensure positive duration
            if silence_duration > 0:
                chord_audio_segments.append(AudioSegment.silent(duration=int(silence_duration)))
            
            # Add synthesized speech with precise timing
            chord_speech = chord_data['speech']
            if chord_speech in tts_cache:
                speech_audio = tts_cache[chord_speech]
                # Limit speech duration to avoid overlap with next chord
                max_duration = 800  # 0.8 seconds max to leave room for next chord
                if len(speech_audio) > max_duration:
                    speech_audio = speech_audio[:max_duration]
                chord_audio_segments.append(speech_audio)
            else:
                # Fallback beep
                beep = AudioSegment.sine(frequency=440, duration=200)
                chord_audio_segments.append(beep)
        
        # Combine chord audio
        chord_track = sum(chord_audio_segments, AudioSegment.empty())
        
        # Step 6: Mix chord vocals with instrumental track
        tasks[task_id]['step'] = 'Overlaying spoken chords onto instrumental track'
        
        instrumental_audio = AudioSegment.from_wav(instrumental_path)
        
        # Ensure chord track matches instrumental audio length
        if len(chord_track) < len(instrumental_audio):
            chord_track += AudioSegment.silent(duration=len(instrumental_audio) - len(chord_track))
        elif len(chord_track) > len(instrumental_audio):
            chord_track = chord_track[:len(instrumental_audio)]
        
        # Mix chord vocals with instrumental (not original audio)
        final_audio = instrumental_audio.overlay(chord_track - 10)  # Reduce chord volume by 10dB
        
        # Export final result
        output_path = os.path.join(task_dir, 'final.mp3')
        final_audio.export(output_path, format='mp3')
        
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['step'] = 'Complete'
        tasks[task_id]['output_file'] = output_path
        
    except Exception as e:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)
        print(f"Processing error for task {task_id}: {e}")

@app.route('/')
def index():
    """Serve the main application page"""
    return send_file('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Get chord type selections (default to all enabled if not provided)
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
    
    if 'chord_types' in request.form:
        try:
            user_chord_types = json.loads(request.form['chord_types'])
            chord_types.update(user_chord_types)
        except (json.JSONDecodeError, TypeError):
            pass  # Use defaults if parsing fails
    
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
        'filename': filename,
        'chord_types': chord_types
    }
    
    # Start background processing
    thread = Thread(target=process_audio_task, args=(task_id, file_path))
    thread.start()
    
    return jsonify({'task_id': task_id, 'status': 'queued'})

@app.route('/status/<task_id>')
def get_status(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    return jsonify(tasks[task_id])

@app.route('/download/<task_id>')
def download_file(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = tasks[task_id]
    if task['status'] != 'completed':
        return jsonify({'error': 'Task not completed'}), 400
    
    return send_file(task['output_file'], as_attachment=True, download_name='chord_vocals.mp3')

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
