#!/usr/bin/env python3
"""
Quick test for dots pronunciation strategy
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import chord_to_ipa_phonemes

# Test the dots strategy
test_chords = ['E', 'B', 'C', 'Em', 'Bm', 'Cm']

print("Testing dots pronunciation strategy:")
print("=" * 40)

for chord in test_chords:
    pronunciation = chord_to_ipa_phonemes(chord)
    print(f"{chord} → '{pronunciation}'")

print("\nExpected results:")
print("E → 'E.'")
print("B → 'B.'") 
print("C → 'C.'")
print("Em → 'E. MINOR'")
print("Bm → 'B. MINOR'")
print("Cm → 'C. MINOR'") 