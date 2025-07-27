#!/usr/bin/env python3
"""
Test script to verify chord detection fixes work with real audio
"""

import os
import sys
import traceback

def test_chord_detection():
    """Test chord detection with a real audio file"""
    
    # Test audio file path
    test_file = "uploads/4843ea20-6e91-4b90-965d-8459612870a3/instrumental_track.wav"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    print(f"✅ Test file found: {test_file}")
    print(f"   File size: {os.path.getsize(test_file) / (1024*1024):.1f} MB")
    
    try:
        # Import our fixed chord detection
        from madmom_compat import detect_chords_simple
        
        print("\n=== Testing chord detection ===")
        print("This may take a few minutes...")
        
        # Run chord detection
        chords = detect_chords_simple(test_file, task_id="test_fix")
        
        if chords and len(chords) > 0:
            print(f"✅ Chord detection succeeded: {len(chords)} chords detected")
            
            # Show first few chords
            print("\nFirst 5 chords:")
            for i, chord in enumerate(chords[:5]):
                print(f"  {i+1}: {chord}")
            
            return True
        else:
            print("❌ No chords detected")
            return False
            
    except Exception as e:
        print(f"❌ Chord detection failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing chord detection fixes...")
    
    success = test_chord_detection()
    
    if success:
        print("\n✅ All tests passed - chord detection is working!")
    else:
        print("\n❌ Tests failed - chord detection still has issues")
        sys.exit(1) 