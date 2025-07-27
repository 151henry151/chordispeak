"""
Madmom compatibility layer for Python 3.11
This module patches madmom to work with Python 3.11 without modifying the package files.
"""

import sys
import collections
import numpy as np

def patch_madmom_for_python311():
    """Apply compatibility patches for madmom to work with Python 3.11"""
    
    # Fix collections.MutableSequence issue
    if not hasattr(collections, 'MutableSequence'):
        from collections.abc import MutableSequence
        collections.MutableSequence = MutableSequence
    
    # Fix numpy type aliases
    if not hasattr(np, 'float'):
        np.float = float
    if not hasattr(np, 'int'):
        np.int = int
    if not hasattr(np, 'complex'):
        np.complex = complex
    
    # Fix numpy scalar types
    if not hasattr(np, 'float32'):
        np.float32 = np.dtype('float32').type
    if not hasattr(np, 'complex64'):
        np.complex64 = np.dtype('complex64').type

def safe_import_madmom():
    """Safely import madmom with compatibility patches applied"""
    try:
        # Apply patches before importing madmom
        patch_madmom_for_python311()
        
        # Now import madmom
        import madmom
        from madmom.features.chords import DeepChromaChordRecognitionProcessor
        
        return madmom, DeepChromaChordRecognitionProcessor
    except Exception as e:
        print(f"Error importing madmom: {e}")
        return None, None

def create_chord_detector():
    """Create a chord detector with proper error handling"""
    madmom, DeepChromaChordRecognitionProcessor = safe_import_madmom()
    
    if madmom is None:
        raise RuntimeError("Failed to import madmom")
    
    try:
        detector = DeepChromaChordRecognitionProcessor()
        return detector
    except Exception as e:
        raise RuntimeError(f"Failed to create chord detector: {e}")

def detect_chords_simple(audio_file, task_id=None):
    """Simple chord detection using madmom's standard approach"""
    print(f"[TASK {task_id}] Starting simple chord detection")
    
    # Apply compatibility patches
    patch_madmom_for_python311()
    
    try:
        # Import madmom with patches applied
        import madmom
        from madmom.features.chords import DeepChromaChordRecognitionProcessor
        
        print(f"[TASK {task_id}] Madmom version: {madmom.__version__}")
        
        # Create detector
        detector = DeepChromaChordRecognitionProcessor()
        print(f"[TASK {task_id}] Chord detector created successfully")
        
        # Process audio file
        print(f"[TASK {task_id}] Processing audio file: {audio_file}")
        chords = detector(audio_file)
        
        print(f"[TASK {task_id}] Chord detection completed: {len(chords)} segments")
        return chords
        
    except Exception as e:
        print(f"[TASK {task_id}] Error in chord detection: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Chord detection failed: {e}")

if __name__ == "__main__":
    # Test the compatibility layer
    print("Testing madmom compatibility...")
    try:
        madmom, detector_class = safe_import_madmom()
        if madmom:
            print(f"✅ Madmom imported successfully: {madmom.__version__}")
        else:
            print("❌ Madmom import failed")
    except Exception as e:
        print(f"❌ Error: {e}") 