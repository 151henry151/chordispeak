#!/usr/bin/env python3
"""
Simple test for dots pronunciation strategy without importing the full app
"""

# Test the dots strategy directly
def chord_to_ipa_phonemes_dots(chord):
    """Convert chord notation to phonetic spellings with dots strategy"""
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
    
    letter_phonemes = letter_phonemes_dots
    
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

# Test the dots strategy
test_chords = ['E', 'B', 'C', 'Em', 'Bm', 'Cm']

print("Testing dots pronunciation strategy:")
print("=" * 40)

for chord in test_chords:
    pronunciation = chord_to_ipa_phonemes_dots(chord)
    print(f"{chord} → '{pronunciation}'")

print("\nExpected results:")
print("E → 'E.'")
print("B → 'B.'") 
print("C → 'C.'")
print("Em → 'E. MINOR'")
print("Bm → 'B. MINOR'")
print("Cm → 'C. MINOR'") 