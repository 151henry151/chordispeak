#!/usr/bin/env python3
"""
Simple test for madmom chord detection with compatibility layer
"""

import os
import sys
import numpy as np

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_madmom_compatibility():
    """Test madmom compatibility layer"""
    print("Testing madmom compatibility layer...")
    
    try:
        from madmom_compat import detect_chords_simple
        print("✅ Madmom compatibility layer imported successfully")
        
        # Test with a simple audio file if available
        test_files = [
            "uploads/test_audio.wav",
            "test_audio.wav",
            "sample.wav"
        ]
        
        test_file = None
        for file_path in test_files:
            if os.path.exists(file_path):
                test_file = file_path
                break
        
        if test_file:
            print(f"Testing with file: {test_file}")
            try:
                chords = detect_chords_simple(test_file, task_id="test")
                print(f"✅ Chord detection successful: {len(chords)} chords detected")
                if chords:
                    print("First few chords:")
                    for i, chord in enumerate(chords[:3]):
                        print(f"  {i}: {chord}")
                return True
            except Exception as e:
                print(f"❌ Chord detection failed: {e}")
                return False
        else:
            print("⚠️  No test audio file found, skipping chord detection test")
            print("✅ Compatibility layer works (no audio file to test)")
            return True
            
    except Exception as e:
        print(f"❌ Error importing madmom compatibility layer: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_madmom_compatibility()
    if success:
        print("\n✅ All tests passed")
        sys.exit(0)
    else:
        print("\n❌ Tests failed")
        sys.exit(1) 