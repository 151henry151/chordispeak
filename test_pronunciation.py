#!/usr/bin/env python3
"""
Test script for different letter pronunciation strategies with Coqui TTS
"""

import os
import sys
import tempfile
import subprocess

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import test_pronunciation_strategies, synthesize_chord_speech_coqui

def test_pronunciation_with_tts():
    """Test different pronunciation strategies with actual TTS synthesis"""
    
    # Test strategies
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
        },
        'Spelled': {
            'A': 'A Y', 'B': 'B E E', 'C': 'C E E', 'D': 'D E E', 'E': 'E E', 'F': 'E F F', 'G': 'G E E'
        }
    }
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing pronunciation strategies in: {temp_dir}")
        
        # We need a voice sample for testing - create a simple one or use existing
        voice_sample_path = None
        
        # Look for existing voice samples
        for root, dirs, files in os.walk('uploads'):
            for file in files:
                if file == 'voice_sample.wav':
                    voice_sample_path = os.path.join(root, file)
                    print(f"Found existing voice sample: {voice_sample_path}")
                    break
            if voice_sample_path:
                break
        
        if not voice_sample_path:
            print("No existing voice sample found. Please run a test first to generate one.")
            return
        
        # Test each strategy
        for strategy_name, letters in strategies.items():
            print(f"\n=== Testing {strategy_name} Strategy ===")
            
            # Test a few key letters
            test_letters = ['E', 'B', 'C']  # The problematic ones
            
            for letter in test_letters:
                pronunciation = letters[letter]
                output_path = os.path.join(temp_dir, f"{strategy_name}_{letter}.wav")
                
                print(f"  Testing {letter} → '{pronunciation}'")
                
                try:
                    success = synthesize_chord_speech_coqui(pronunciation, voice_sample_path, output_path)
                    if success and os.path.exists(output_path):
                        file_size = os.path.getsize(output_path)
                        print(f"    ✓ Generated: {output_path} ({file_size} bytes)")
                    else:
                        print(f"    ✗ Failed to generate audio")
                except Exception as e:
                    print(f"    ✗ Error: {e}")
        
        print(f"\n=== Test Complete ===")
        print(f"Check the generated files in: {temp_dir}")
        print("You can play them to compare pronunciation quality.")

if __name__ == "__main__":
    print("Letter Pronunciation Test for Coqui TTS")
    print("=" * 50)
    
    # First show all strategies
    test_pronunciation_strategies()
    
    # Then test with actual TTS
    test_pronunciation_with_tts() 