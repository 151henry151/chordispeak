"""
Madmom compatibility layer for Python 3.11
This module provides a clean implementation following the madmom documentation exactly.
"""

import sys
import collections
import numpy as np
import os

# Apply patches immediately when module is imported
def _apply_immediate_patches():
    """Apply patches that need to be done before any madmom imports"""
    
    # Fix collections.MutableSequence issue
    if not hasattr(collections, 'MutableSequence'):
        from collections.abc import MutableSequence
        collections.MutableSequence = MutableSequence
    
    # Fix numpy type aliases for newer numpy versions
    if not hasattr(np, 'float'):
        np.float = float
    if not hasattr(np, 'int'):
        np.int = int
    if not hasattr(np, 'complex'):
        np.complex = complex

# Apply patches immediately
_apply_immediate_patches()

def detect_chords_official_approach(audio_file, task_id=None):
    """
    Chord detection using the exact approach from the official DCChordRecognition tool.
    
    This follows the two-step process used by the official madmom command-line tool:
    1. DeepChromaProcessor - Extract chroma features
    2. DeepChromaChordRecognitionProcessor - Recognize chords from chroma features
    
    Based on the official implementation at:
    https://github.com/CPJKU/madmom/blob/main/bin/DCChordRecognition
    """
    print(f"[TASK {task_id}] Starting official DCChordRecognition approach")
    
    try:
        # Import madmom modules following the official tool
        import madmom
        from madmom.audio import DeepChromaProcessor
        from madmom.features import DeepChromaChordRecognitionProcessor
        
        print(f"[TASK {task_id}] Madmom version: {madmom.__version__}")
        
        # Step 1: Extract chroma features using DeepChromaProcessor
        print(f"[TASK {task_id}] Step 1: Extracting chroma features with DeepChromaProcessor...")
        chroma_processor = DeepChromaProcessor(fps=10)  # Use same fps as official tool
        chroma_features = chroma_processor(audio_file)
        print(f"[TASK {task_id}] Chroma features extracted: shape={chroma_features.shape}")
        
        # Step 2: Recognize chords using DeepChromaChordRecognitionProcessor
        print(f"[TASK {task_id}] Step 2: Recognizing chords with DeepChromaChordRecognitionProcessor...")
        chord_processor = DeepChromaChordRecognitionProcessor()
        chords = chord_processor(chroma_features)
        
        print(f"[TASK {task_id}] Official approach completed: {len(chords)} segments")
        
        # Process results - handle both array and list formats
        if chords is not None:
            # Convert to list if it's a numpy array
            if hasattr(chords, 'shape'):
                print(f"[TASK {task_id}] Converting numpy array to list format")
                # Handle numpy array format
                chord_list = []
                for i in range(len(chords)):
                    if len(chords[i]) >= 2:
                        chord_list.append([float(chords[i][0]), float(chords[i][1]), str(chords[i][2])])
                chords = chord_list
            
            # Check if we have valid chords
            if isinstance(chords, (list, tuple)) and len(chords) > 0:
                print(f"[TASK {task_id}] Successfully detected {len(chords)} chord segments")
                
                # Show first few chords for debugging
                print(f"[TASK {task_id}] First 5 chords:")
                for i, chord in enumerate(chords[:5]):
                    if len(chord) >= 2:
                        print(f"[TASK {task_id}]   {i+1}: Time: {chord[0]:.2f}, Chord: {chord[1]}")
                    else:
                        print(f"[TASK {task_id}]   {i+1}: {chord}")
                
                return chords
            else:
                print(f"[TASK {task_id}] No valid chords detected in results")
                return []
        else:
            print(f"[TASK {task_id}] No chords detected")
            return []
        
    except Exception as e:
        print(f"[TASK {task_id}] Error in official approach: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to chroma-based method if official approach fails
        print(f"[TASK {task_id}] Official approach failed, trying chroma-based fallback...")
        return detect_chords_chroma_fallback(audio_file, task_id)

def detect_chords_simple(audio_file, task_id=None):
    """
    Chord detection using the official DCChordRecognition approach.
    
    This follows the two-step process used by the official madmom command-line tool:
    1. DeepChromaProcessor - Extract chroma features
    2. DeepChromaChordRecognitionProcessor - Recognize chords from chroma features
    
    Based on the official implementation at:
    https://github.com/CPJKU/madmom/blob/main/bin/DCChordRecognition
    
    Args:
        audio_file (str): Path to the audio file
        task_id (str): Task ID for logging
        
    Returns:
        list: List of chord detections with timing information
    """
    print(f"[TASK {task_id}] Starting official DCChordRecognition approach")
    
    try:
        # Import madmom modules following the official tool
        import madmom
        from madmom.audio import DeepChromaProcessor
        from madmom.features import DeepChromaChordRecognitionProcessor
        
        print(f"[TASK {task_id}] Madmom version: {madmom.__version__}")
        
        # Step 1: Extract chroma features with DeepChromaProcessor
        print(f"[TASK {task_id}] Step 1: Extracting chroma features with DeepChromaProcessor...")
        chroma_processor = DeepChromaProcessor()
        chroma_features = chroma_processor(audio_file)
        print(f"[TASK {task_id}] Chroma features extracted: shape={chroma_features.shape}")
        
        # Step 2: Recognize chords from chroma features  
        print(f"[TASK {task_id}] Step 2: Recognizing chords with DeepChromaChordRecognitionProcessor...")
        chord_processor = DeepChromaChordRecognitionProcessor()
        chords = chord_processor(chroma_features)
        print(f"[TASK {task_id}] Official approach completed: {len(chords)} segments")
        
        # Step 3: Convert results to expected format
        if hasattr(chords, 'tolist'):
            print(f"[TASK {task_id}] Converting numpy array to list format")
            chords = chords.tolist()
        
        if chords and len(chords) > 0:
            print(f"[TASK {task_id}] Successfully detected {len(chords)} chord segments")
            
            # Show first few chords for debugging
            print(f"[TASK {task_id}] First 5 chords:")
            for i, chord in enumerate(chords[:5]):
                if len(chord) >= 2:
                    print(f"[TASK {task_id}]   {i+1}: Time: {chord[0]:.2f}, Chord: {chord[1]}")
                else:
                    print(f"[TASK {task_id}]   {i+1}: {chord}")
            
            return chords
        else:
            print(f"[TASK {task_id}] No chords detected")
            return []
        
    except Exception as e:
        print(f"[TASK {task_id}] Error in official DCChordRecognition approach: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"DCChordRecognition failed: {e}")

def detect_chords_chroma_fallback(audio_file, task_id=None):
    """
    Alternative chord detection using madmom's chroma features without CRF models.
    
    This approach uses madmom's chroma feature extraction and implements
    a simple chord recognition algorithm to avoid the problematic CRF models.
    """
    print(f"[TASK {task_id}] Starting chroma-based chord detection fallback")
    
    try:
        # Import madmom modules
        import madmom
        from madmom.audio.chroma import DeepChromaProcessor
        
        print(f"[TASK {task_id}] Madmom version: {madmom.__version__}")
        
        # Step 1: Extract chroma features using madmom's chroma processor
        print(f"[TASK {task_id}] Extracting chroma features...")
        
        try:
            chroma_processor = DeepChromaProcessor()
            chroma_features = chroma_processor(audio_file)
            print(f"[TASK {task_id}] DeepChromaProcessor succeeded: shape={chroma_features.shape}")
        except Exception as e:
            print(f"[TASK {task_id}] DeepChromaProcessor failed: {e}")
            raise RuntimeError("Chroma processor failed")
        
        # Step 2: Implement simple chord recognition from chroma features
        print(f"[TASK {task_id}] Implementing chord recognition from chroma features...")
        
        # Define chord templates for major and minor chords
        chord_templates = {
            'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # C major
            'Cm': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # C minor
            'D': [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],   # D major
            'Dm': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # D minor
            'E': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],   # E major
            'Em': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],  # E minor
            'F': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],   # F major
            'Fm': [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # F minor
            'G': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],   # G major
            'Gm': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # G minor
            'A': [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],   # A major
            'Am': [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # A minor
            'B': [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],   # B major
            'Bm': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # B minor
        }
        
        # Convert templates to numpy arrays
        for chord_name, template in chord_templates.items():
            chord_templates[chord_name] = np.array(template, dtype=np.float32)
        
        # Process chroma features to detect chords
        chords = []
        frame_duration = 0.1  # 100ms frames
        min_chord_duration = 0.5  # Minimum chord duration
        
        for i in range(chroma_features.shape[0]):
            chroma_frame = chroma_features[i, :12]  # Use first 12 dimensions (chroma)
            
            # Normalize chroma frame
            if np.sum(chroma_frame) > 0:
                chroma_frame = chroma_frame / np.sum(chroma_frame)
            
            # Find best matching chord
            best_chord = None
            best_score = 0
            
            for chord_name, template in chord_templates.items():
                # Calculate correlation between chroma frame and chord template
                correlation = np.corrcoef(chroma_frame, template)[0, 1]
                if not np.isnan(correlation) and correlation > best_score:
                    best_score = correlation
                    best_chord = chord_name
            
            # Only add chord if correlation is high enough
            if best_chord and best_score > 0.3:
                time = i * frame_duration
                chords.append([time, best_chord, best_score])
        
        # Apply smoothing and filtering
        filtered_chords = []
        if chords:
            current_chord = chords[0]
            current_start = current_chord[0]
            
            for chord in chords[1:]:
                if chord[1] != current_chord[1] or (chord[0] - current_start) > 2.0:
                    # End current chord
                    if chord[0] - current_start >= min_chord_duration:
                        filtered_chords.append([current_start, chord[0], current_chord[1]])
                    
                    # Start new chord
                    current_chord = chord
                    current_start = chord[0]
            
            # Add final chord
            if len(chords) > 0:
                final_time = chords[-1][0] + frame_duration
                if final_time - current_start >= min_chord_duration:
                    filtered_chords.append([current_start, final_time, current_chord[1]])
        
        print(f"[TASK {task_id}] Chroma-based chord detection completed: {len(filtered_chords)} segments")
        
        # Convert to madmom format
        result = []
        for start_time, end_time, chord_name in filtered_chords:
            result.append([start_time, end_time, chord_name, 0.8])  # Default confidence
        
        return result
        
    except Exception as e:
        print(f"[TASK {task_id}] Error in chroma-based chord detection: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Chroma-based chord detection failed: {e}")

def detect_chords_advanced(audio_file, task_id=None):
    """
    Advanced chord detection using madmom with two-step approach.
    
    This follows the madmom documentation for advanced usage with separate
    chroma feature extraction and chord recognition steps.
    """
    print(f"[TASK {task_id}] Starting advanced madmom chord detection")
    
    try:
        # Import madmom modules
        import madmom
        from madmom.features.chords import DeepChromaProcessor, DeepChromaChordRecognitionProcessor
        
        print(f"[TASK {task_id}] Madmom version: {madmom.__version__}")
        
        # Step 1: Chroma feature extraction
        print(f"[TASK {task_id}] Step 1: Chroma feature extraction...")
        chroma_processor = DeepChromaProcessor()
        chroma_features = chroma_processor(audio_file)
        
        # Step 2: Chord recognition from chroma features
        print(f"[TASK {task_id}] Step 2: Chord recognition...")
        chord_processor = DeepChromaChordRecognitionProcessor()
        chords = chord_processor(chroma_features)
        
        print(f"[TASK {task_id}] Advanced chord detection completed: {len(chords)} segments")
        return chords
        
    except Exception as e:
        print(f"[TASK {task_id}] Error in advanced chord detection: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Advanced chord detection failed: {e}")

if __name__ == "__main__":
    # Test the compatibility layer
    print("Testing madmom compatibility...")
    try:
        import madmom
        print(f"✅ Madmom imported successfully: {madmom.__version__}")
        
        # Test basic functionality
        from madmom.features.chords import DeepChromaChordRecognitionProcessor
        detector = DeepChromaChordRecognitionProcessor()
        print("✅ Chord detector created successfully")
        
    except Exception as e:
        print(f"❌ Error: {e}") 