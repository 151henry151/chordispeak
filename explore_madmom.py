#!/usr/bin/env python3
"""
Explore madmom API to find working chord detection methods
"""

# Apply patches first
import collections
if not hasattr(collections, 'MutableSequence'):
    from collections.abc import MutableSequence
    collections.MutableSequence = MutableSequence

import numpy as np
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'complex'):
    np.complex = complex

def explore_madmom():
    """Explore what's available in madmom"""
    print("Exploring madmom API...")
    
    try:
        import madmom
        print(f"✅ Madmom imported: {madmom.__version__}")
        
        # Explore features module
        print("\n=== Exploring madmom.features ===")
        import madmom.features
        print(f"Features modules: {dir(madmom.features)}")
        
        # Explore chords module
        print("\n=== Exploring madmom.features.chords ===")
        import madmom.features.chords
        print(f"Chord processors: {dir(madmom.features.chords)}")
        
        # Explore audio module
        print("\n=== Exploring madmom.audio ===")
        import madmom.audio
        print(f"Audio modules: {dir(madmom.audio)}")
        
        # Try to import specific processors
        print("\n=== Testing specific imports ===")
        
        try:
            from madmom.features.chords import DeepChromaChordRecognitionProcessor
            print("✅ DeepChromaChordRecognitionProcessor imported")
        except Exception as e:
            print(f"❌ DeepChromaChordRecognitionProcessor failed: {e}")
        
        try:
            from madmom.audio.chroma import DeepChromaProcessor
            print("✅ DeepChromaProcessor imported")
        except Exception as e:
            print(f"❌ DeepChromaProcessor failed: {e}")
        
        try:
            from madmom.audio.chroma import ChromaProcessor
            print("✅ ChromaProcessor imported")
        except Exception as e:
            print(f"❌ ChromaProcessor failed: {e}")
        
        # Test basic audio loading
        print("\n=== Testing audio loading ===")
        try:
            from madmom.io.audio import load_audio_file
            print("✅ load_audio_file imported")
        except Exception as e:
            print(f"❌ load_audio_file failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error exploring madmom: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    explore_madmom() 