#!/usr/bin/env python3
"""
Test script to isolate chord detection issues
"""

import os
import sys
import traceback

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_chord_detection():
    """Test chord detection with detailed error reporting"""
    
    # Import dependencies
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
        
        import librosa
        print(f"Librosa version: {librosa.__version__}")
        
        import madmom
        print(f"Madmom version: {madmom.__version__}")
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    
    # Test audio file path
    test_file = "uploads/14cf72c1-912c-4758-b591-14a0059a7794/instrumental_track.wav"
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return False
    
    print(f"Testing chord detection on: {test_file}")
    print(f"File size: {os.path.getsize(test_file)} bytes")
    
    # Test 1: Load audio with librosa
    try:
        print("\n=== Test 1: Loading audio with librosa ===")
        y, sr = librosa.load(test_file, sr=None, mono=True, dtype=np.float32)
        print(f"Audio loaded: shape={y.shape}, dtype={y.dtype}, sr={sr}")
        print(f"Audio range: min={np.min(y)}, max={np.max(y)}")
        print(f"Audio has NaN: {np.any(np.isnan(y))}")
        print(f"Audio has Inf: {np.any(np.isinf(y))}")
        
        # Check for string-like data
        if y.dtype.kind in ['U', 'S']:
            print("ERROR: Audio data contains strings!")
            return False
            
    except Exception as e:
        print(f"Error loading audio: {e}")
        traceback.print_exc()
        return False
    
    # Test 2: Initialize madmom processor
    try:
        print("\n=== Test 2: Initializing madmom processor ===")
        from madmom.features.chords import DeepChromaChordRecognitionProcessor
        
        # Apply NumPy compatibility fixes
        if not hasattr(np, 'float'):
            np.float = float
        if not hasattr(np, 'int'):
            np.int = int
        if not hasattr(np, 'complex'):
            np.complex = complex
        
        chord_detector = DeepChromaChordRecognitionProcessor()
        print("Madmom processor initialized successfully")
        
    except Exception as e:
        print(f"Error initializing madmom processor: {e}")
        traceback.print_exc()
        return False
    
    # Test 3: Try chord detection
    try:
        print("\n=== Test 3: Running chord detection ===")
        
        # Try with madmom's own signal loader first
        try:
            from madmom.audio.signal import Signal
            signal = Signal(test_file)
            print(f"Madmom signal loaded: shape={signal.shape}, dtype={signal.dtype}")
            chords = chord_detector(signal)
            print(f"Chord detection with Signal succeeded: {len(chords)} chords")
        except Exception as signal_error:
            print(f"Madmom signal loading failed: {signal_error}")
            # Fall back to file path
            chords = chord_detector(test_file)
            print(f"Chord detection with file path succeeded: {len(chords)} chords")
        
        # Print first few chords
        if chords and len(chords) > 0:
            print("First 5 chords:")
            for i, chord in enumerate(chords[:5]):
                print(f"  {i}: {chord}")
        
        return True
        
    except Exception as e:
        print(f"Error during chord detection: {e}")
        traceback.print_exc()
        
        # Check for specific NumPy errors
        if "ufunc 'multiply'" in str(e):
            print("\n=== NumPy Type Mismatch Analysis ===")
            print("This appears to be a NumPy type mismatch error.")
            
            # Check if it's a string data issue
            if "dtype('<U1')" in str(e) or "dtype('<U32')" in str(e):
                print("ERROR: String data detected in processing!")
                print("This suggests madmom is receiving string data instead of numeric data.")
                
                # Try to reload and verify audio data
                try:
                    y2, sr2 = librosa.load(test_file, sr=None, mono=True, dtype=np.float32)
                    print(f"Reloaded audio: shape={y2.shape}, dtype={y2.dtype}")
                    
                    if not np.issubdtype(y2.dtype, np.number):
                        print(f"ERROR: Audio data is not numeric! dtype={y2.dtype}")
                    else:
                        print("Audio data is numeric, issue might be in madmom processing")
                        
                except Exception as reload_error:
                    print(f"Error reloading audio: {reload_error}")
        
        return False

if __name__ == "__main__":
    print("=== Chord Detection Test ===")
    success = test_chord_detection()
    if success:
        print("\n✅ Test completed successfully")
    else:
        print("\n❌ Test failed")
    sys.exit(0 if success else 1) 