        
        # Restore original torch.load
        torch.load = original_torch_load
        return True
    except Exception as e:
        print(f"Coqui TTS synthesis error: {e}")
        return False

def synthesize_chord_speech(text, voice_sample_path, output_path):
    """Generate speech using only Coqui XTTS v2 voice cloning"""
    if not lazy_import_tts():
        raise RuntimeError("Coqui TTS not available. Please install Coqui TTS (XTTS v2).")
    if not voice_sample_path or not os.path.exists(voice_sample_path):
        raise RuntimeError("Voice sample for cloning not found. Cannot synthesize without a reference voice.")
    return synthesize_chord_speech_coqui(text, voice_sample_path, output_path)

def detect_chords(audio_file, chord_types=None, task_id=None):
    """Detect chords from audio file using madmom for accurate chord recognition with progress tracking"""
    print(f"[TASK {task_id}] Starting detect_chords function")
    
    if not lazy_import_audio_deps():
        raise RuntimeError("Audio processing dependencies not available. Cannot perform chord detection.")
    
    # Comprehensive audio file validation
    print(f"[TASK {task_id}] Validating audio file: {audio_file}")
    try:
        import os
        if not os.path.exists(audio_file):
            raise RuntimeError(f"Audio file does not exist: {audio_file}")
        
        file_size = os.path.getsize(audio_file)
        if file_size == 0:
            raise RuntimeError(f"Audio file is empty: {audio_file}")
        
        print(f"[TASK {task_id}] Audio file validation passed: size={file_size} bytes")
        
        # Try to validate with soundfile first
        import soundfile as sf
        try:
            info = sf.info(audio_file)
            print(f"[TASK {task_id}] Soundfile validation: samplerate={info.samplerate}, channels={info.channels}, duration={info.duration:.2f}s")
            if info.duration <= 0:
                raise RuntimeError(f"Audio file has zero duration: {audio_file}")
        except Exception as sf_error:
            print(f"[TASK {task_id}] Warning: Soundfile validation failed: {sf_error}")
            # Continue with librosa validation
        
    except Exception as validation_error:
        print(f"[TASK {task_id}] Audio file validation failed: {validation_error}")
        raise RuntimeError(f"Audio file validation failed: {validation_error}")
        
    try:
        import time
        print(f"[TASK {task_id}] Importing madmom modules...")
        from madmom.features.chords import DeepChromaChordRecognitionProcessor
        print(f"[TASK {task_id}] Madmom modules imported successfully")
        
        # Check what processors are actually available
        try:
            from madmom.features.chords import DeepChromaProcessor
            print(f"[TASK {task_id}] DeepChromaProcessor is available")
            chroma_processor_available = True
        except ImportError:
            print(f"[TASK {task_id}] DeepChromaProcessor not available, using single-step approach")
            chroma_processor_available = False
        
        # Debug: Print audio file info
        import soundfile as sf
        try:
            info = sf.info(audio_file)
            print(f"[TASK {task_id}] Audio file info: samplerate={info.samplerate}, channels={info.channels}, duration={info.duration:.2f}s, format={info.format}, subtype={info.subtype}")
        except Exception as e:
            print(f"[TASK {task_id}] Could not get audio file info with soundfile: {e}")
        
        # Get audio duration for progress estimation
        # Use madmom's own audio loading for better compatibility
        print(f"[TASK {task_id}] Loading audio with madmom's audio loader...")
        try:
            from madmom.io.audio import load_audio_file
            y, sr = load_audio_file(audio_file, sample_rate=44100, num_channels=1, dtype=np.float32)
            audio_duration = len(y) / sr
            print(f"[TASK {task_id}] Madmom audio loader: duration={audio_duration:.2f}s, sr={sr}, shape={y.shape}, dtype={y.dtype}")
        except Exception as madmom_load_error:
            print(f"[TASK {task_id}] Madmom audio loading failed: {madmom_load_error}")
            print(f"[TASK {task_id}] Falling back to librosa...")
            # Fallback to librosa
            y, sr = librosa.load(audio_file, sr=None, mono=True, dtype=np.float32)
            audio_duration = len(y) / sr
            print(f"[TASK {task_id}] Librosa fallback: duration={audio_duration:.2f}s, sr={sr}, shape={y.shape}, dtype={y.dtype}")
        
        # CRITICAL: Check for string data in audio (this causes the NumPy error)
        if y.dtype.kind in ['U', 'S']:
            print(f"[TASK {task_id}] ERROR: Audio data contains strings! dtype={y.dtype}")
            print(f"[TASK {task_id}] This will cause the NumPy multiply error. Attempting to fix...")
            
            # Try to reload with different parameters
            try:
                print(f"[TASK {task_id}] Attempting to reload audio with explicit numeric conversion...")
                y, sr = librosa.load(audio_file, sr=44100, mono=True, dtype=np.float32)
                
                # Check again
                if y.dtype.kind in ['U', 'S']:
                    print(f"[TASK {task_id}] ERROR: Still getting string data after reload!")
                    raise RuntimeError("Audio file contains string data instead of numeric data")
                else:
                    print(f"[TASK {task_id}] Successfully reloaded audio with numeric data: dtype={y.dtype}")
            except Exception as reload_error:
                print(f"[TASK {task_id}] Failed to reload audio: {reload_error}")
                raise RuntimeError(f"Audio file appears to be corrupted or contains invalid data: {reload_error}")
        
        # Additional validation: Check for NaN or Inf values
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            print(f"[TASK {task_id}] WARNING: Audio contains NaN or Inf values, cleaning...")
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Ensure audio data is float32 for madmom compatibility
        if y.dtype != np.float32:
            print(f"[TASK {task_id}] Converting audio dtype from {y.dtype} to float32")
            y = y.astype(np.float32)
        
        # Additional NumPy compatibility fixes for madmom
        print(f"[TASK {task_id}] Applying NumPy compatibility fixes...")
        if not hasattr(np, 'float'):
            np.float = float
        if not hasattr(np, 'int'):
            np.int = int
        if not hasattr(np, 'complex'):
            np.complex = complex
        
        # Ensure audio is properly formatted for madmom
        print(f"[TASK {task_id}] Preparing audio for madmom processing...")
        # Normalize audio to prevent overflow issues
        if np.max(np.abs(y)) > 1.0:
            y = y / np.max(np.abs(y))
        
        # Ensure audio is contiguous in memory
        y = np.ascontiguousarray(y, dtype=np.float32)
        
        # Final validation before processing
        print(f"[TASK {task_id}] Final audio validation:")
        print(f"[TASK {task_id}]   - Shape: {y.shape}")
        print(f"[TASK {task_id}]   - Dtype: {y.dtype}")
        print(f"[TASK {task_id}]   - Range: [{np.min(y):.6f}, {np.max(y):.6f}]")
        print(f"[TASK {task_id}]   - Has NaN: {np.any(np.isnan(y))}")
        print(f"[TASK {task_id}]   - Has Inf: {np.any(np.isinf(y))}")
        print(f"[TASK {task_id}]   - Is numeric: {np.issubdtype(y.dtype, np.number)}")
        
        if not np.issubdtype(y.dtype, np.number):
            raise RuntimeError("Audio data is not numeric - cannot proceed with chord detection")
        
        # Initialize madmom chord detection using the recommended two-step approach
        print(f"[TASK {task_id}] Initializing madmom processors...")
        try:
            # Add debug info about madmom version and dependencies
            import madmom
            print(f"[TASK {task_id}] Madmom version: {madmom.__version__}")
            
            # Check NumPy version compatibility
            print(f"[TASK {task_id}] NumPy version: {np.__version__}")
            print(f"[TASK {task_id}] Librosa version: {librosa.__version__}")
            
            # Check for known compatibility issues
            if hasattr(madmom, '__version__') and madmom.__version__:
                print(f"[TASK {task_id}] Madmom version check completed")
            else:
                print(f"[TASK {task_id}] Warning: Could not determine madmom version")
            
            # Initialize madmom processors with recommended parameters
            # Step 1: Chroma feature extraction
            if chroma_processor_available:
                chroma_processor = DeepChromaProcessor(
                    sample_rate=44100,  # Standard sample rate
                    hop_size=512,       # Frame hop size for good temporal resolution
                    fps=50,            # 50 frames per second for smooth detection
                    num_classes=25      # Standard number of chord classes
                )
                print(f"[TASK {task_id}] Chroma processor initialized successfully")
            else:
                print(f"[TASK {task_id}] DeepChromaProcessor not available, using single-step approach")
            
            # Step 2: Chord recognition from chroma features
            chord_processor = DeepChromaChordRecognitionProcessor()
            print(f"[TASK {task_id}] Chord processor initialized successfully")
            
        except Exception as init_error:
            print(f"[TASK {task_id}] ERROR initializing madmom processors: {init_error}")
            # Fallback to single-step approach
            print(f"[TASK {task_id}] Falling back to single-step approach...")
            chord_detector = DeepChromaChordRecognitionProcessor()
            print(f"[TASK {task_id}] Fallback processor initialized successfully")
            
        except Exception as e:
            print(f"[TASK {task_id}] ERROR initializing madmom processor: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize madmom processor: {e}")
        
        # Start timing for progress estimation
        start_time = time.time()
        
        # Initialize variables for chord processing with more conservative thresholds
        filtered_out_count = 0
        min_confidence = 0.5  # More conservative confidence threshold
        min_chord_duration = 0.5  # Longer minimum chord duration
        min_time_between_chords = 1.0  # More realistic for speech synthesis
        valid_chords = []
        
        # Update progress: Starting chord detection (40-50%)
        if task_id and task_id in tasks:
            tasks[task_id]['step'] = 'Analyzing chord pattern'
            tasks[task_id]['progress'] = 40
            print(f"[TASK {task_id}] Progress: 40% - Starting chord detection")
        
        # Process the audio file with the chord detector
        print(f"[TASK {task_id}] Starting chord detection...")
        try:
            print(f"[TASK {task_id}] Calling chord detection with audio_file: {audio_file}")
            # Add debug info about the audio file
            import os
            if os.path.exists(audio_file):
                file_size = os.path.getsize(audio_file)
                print(f"[TASK {task_id}] Audio file size: {file_size} bytes")
                
                # Check if file is not empty
                if file_size == 0:
                    print(f"[TASK {task_id}] ERROR: Audio file is empty!")
                    raise RuntimeError("Audio file is empty")
                
                # Try to validate the audio file format
                try:
                    import soundfile as sf
                    info = sf.info(audio_file)
                    print(f"[TASK {task_id}] Audio file format: {info.format}, subtype: {info.subtype}")
                    if info.duration <= 0:
                        print(f"[TASK {task_id}] ERROR: Audio file has zero duration!")
                        raise RuntimeError("Audio file has zero duration")
                except Exception as sf_error:
                    print(f"[TASK {task_id}] Warning: Could not validate audio file with soundfile: {sf_error}")
            else:
                print(f"[TASK {task_id}] ERROR: Audio file does not exist: {audio_file}")
                raise RuntimeError(f"Audio file not found: {audio_file}")
            
            try:
                # Debug: Check what madmom is receiving
                print(f"[TASK {task_id}] Audio file path: {audio_file}")
                print(f"[TASK {task_id}] Audio file type: {type(audio_file)}")
                
                # Use the main chord detection approach
                print(f"[TASK {task_id}] Using main chord detection approach...")
                chords = chord_detector(audio_file)
                print(f"[TASK {task_id}] Chord detection completed successfully")
                
            except Exception as chord_error:
                print(f"[TASK {task_id}] ERROR during chord detection call: {chord_error}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Chord detection failed: {chord_error}")
                
        except Exception as e:
            print(f"[TASK {task_id}] ERROR in chord detection: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Chord detection failed: {e}")
        
        # Save chord data
        chords_file = os.path.join(task_dir, 'chords.json')
        with open(chords_file, 'w') as f:
            json.dump(chords, f)
        print(f"Chord data saved: {chords_file}")
        
        # Step 5: Voice synthesis using voice cloning
        print(f"\n=== [TASK {task_id}] STEP 5: VOICE SYNTHESIS ===")
        tasks[task_id]['step'] = 'Synthesizing spoken chord overlay'
        tasks[task_id]['progress'] = 70
        print(f"[TASK {task_id}] Progress: 70% - Starting voice synthesis")
        
        tts_start = time.time()
        unique_chords = list(set(chord_data['speech'] for chord_data in chords))
        print(f"[TASK {task_id}] Unique chords to synthesize: {len(unique_chords)}")
        print(f"[TASK {task_id}] Unique chords: {unique_chords}")
        
        tts_cache = {}
        
        # Update progress for each chord synthesis
        for i, chord_speech in enumerate(unique_chords):
            # Update progress for each chord (70-85%) with whole numbers only
            # Simple mapping: 0->70%, 1->72%, 2->75%, 3->77%, 4->80%, 5->82%, 6->85%
            if len(unique_chords) == 1:
                chord_progress = 70
            elif len(unique_chords) == 2:
                chord_progress = 70 if i == 0 else 85
            elif len(unique_chords) == 3:
                chord_progress = 70 if i == 0 else (77 if i == 1 else 85)
            elif len(unique_chords) == 4:
                chord_progress = 70 if i == 0 else (75 if i == 1 else (80 if i == 2 else 85))
            else:
                # For 5+ chords, use simple increments
                chord_progress = 70 + (i * 3)  # 70, 73, 76, 79, 82, 85
                if chord_progress > 85:
                    chord_progress = 85
            
            tasks[task_id]['progress'] = chord_progress
            tasks[task_id]['step'] = f'Synthesizing chord {i+1}/{len(unique_chords)}'
            print(f"[TASK {task_id}] Progress: {chord_progress}% - Synthesizing chord {i+1}/{len(unique_chords)}: {chord_speech}")
            
            tts_output_path = os.path.join(task_dir, f'tts_{chord_speech.replace(" ", "_").replace("#", "sharp")}.wav')
            if not synthesize_chord_speech(chord_speech, voice_sample_path, tts_output_path):
                error_msg = f"TTS synthesis failed for chord: {chord_speech}"
                print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)
            if os.path.exists(tts_output_path):
                tts_cache[chord_speech] = AudioSegment.from_wav(tts_output_path)
                print(f"Loaded TTS for '{chord_speech}': {len(tts_cache[chord_speech])}ms duration")
            else:
                error_msg = f"TTS output file not created for chord: {chord_speech}"
                print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)
        
        # Step 6: Creating chord audio track
        print(f"Step 6: Creating chord audio track for task {task_id}")
        tasks[task_id]['step'] = 'Creating chord audio track'
        tasks[task_id]['progress'] = 85
        print(f"[TASK {task_id}] Progress: 85% - Creating chord audio track")
        chord_audio_segments = []
        for i, chord_data in enumerate(chords):
            if i == 0:
                silence_duration = chord_data['time'] * 1000
            else:
                silence_duration = (chord_data['time'] - chords[i-1]['time']) * 1000
            if silence_duration > 0:
                chord_audio_segments.append(AudioSegment.silent(duration=int(silence_duration)))
                print(f"Added {int(silence_duration)}ms silence before chord {i+1}")
            chord_speech = chord_data['speech']
            if chord_speech in tts_cache:
                speech_audio = tts_cache[chord_speech]
                chord_audio_segments.append(speech_audio)
                print(f"Added '{chord_speech}' at {chord_data['time']:.2f}s: {len(speech_audio)}ms duration")
            else:
                beep = AudioSegment.sine(frequency=440, duration=200)
                chord_audio_segments.append(beep)
                print(f"Added fallback beep for '{chord_speech}' at {chord_data['time']:.2f}s")
        chord_track = sum(chord_audio_segments, AudioSegment.empty())
        
        # Step 7: Mixing final audio
        print(f"Step 7: Mixing final audio for task {task_id}")
        tasks[task_id]['step'] = 'Overlaying spoken chords onto instrumental track'
        tasks[task_id]['progress'] = 90
        print(f"[TASK {task_id}] Progress: 90% - Mixing final audio")
        instrumental_audio = AudioSegment.from_wav(instrumental_path)
        if len(chord_track) < len(instrumental_audio):
            chord_track += AudioSegment.silent(duration=len(instrumental_audio) - len(chord_track))
        elif len(chord_track) > len(instrumental_audio):
            chord_track = chord_track[:len(instrumental_audio)]
        final_audio = instrumental_audio.overlay(chord_track - 10)
        output_path = os.path.join(task_dir, 'final.mp3')
        final_audio.export(output_path, format='mp3')
        
        # Complete
        total_time = time.time() - start_time
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['step'] = 'Complete'
        tasks[task_id]['progress'] = 100
        tasks[task_id]['output_file'] = output_path
        print(f"\n=== [TASK {task_id}] PROCESSING COMPLETED ===")
        print(f"[TASK {task_id}] Progress: 100% - Processing completed successfully")
        print(f"[TASK {task_id}] Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"[TASK {task_id}] Output file: {output_path}")
        
    except Exception as e:
        print(f"\n=== [TASK {task_id}] PROCESSING ERROR ===")
        print(f"[TASK {task_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)
        tasks[task_id]['step'] = f'Error: {str(e)}'
        print(f"[TASK {task_id}] Processing error for task {task_id}: {e}")

@app.route('/')
def index():
    """Serve the main application page"""
    try:
        return send_file('index.html')
    except Exception as e:
        print(f"Error serving index.html: {e}")
        return f"Error: {str(e)}", 500

@app.route('/Logo.png')
def serve_logo():
    """Serve the logo image"""
    try:
        return send_file('Logo-transparent.png', mimetype='image/png')
    except Exception as e:
        print(f"Error serving Logo-transparent.png: {e}")
        return f"Error: {str(e)}", 500

@app.route('/Logo-transparent.png')
def serve_logo_transparent():
    """Serve the transparent logo image"""
    try:
        return send_file('Logo-transparent.png', mimetype='image/png')
    except Exception as e:
        print(f"Error serving Logo-transparent.png: {e}")
        return f"Error: {str(e)}", 500

@app.route('/favicon.ico')
def serve_favicon_ico():
    """Serve the favicon.ico file"""
    try:
        return send_file('favicon.ico', mimetype='image/x-icon')
    except Exception as e:
        print(f"Error serving favicon.ico: {e}")
        return f"Error: {str(e)}", 500

@app.route('/favicon.png')
def serve_favicon_png():
    """Serve the favicon.png file"""
    try:
        return send_file('favicon.png', mimetype='image/png')
    except Exception as e:
        print(f"Error serving favicon.png: {e}")
        return f"Error: {str(e)}", 500

@app.route('/test')
def test():
    """Simple test route to verify the app is working"""
    return jsonify({
        'status': 'ok',
        'message': 'Flask app is running',
        'version': VERSION
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Remove chord type selections (always use madmom's default)
    
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
        'filename': filename
    }
    
    # Start background processing
    thread = Thread(target=process_audio_task, args=(task_id, file_path))
    thread.start()
    
    return jsonify({'task_id': task_id, 'status': 'queued'})

@app.route('/status/<task_id>')
def get_status(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    # Don't log status requests to reduce terminal noise
    return jsonify(tasks[task_id])

@app.route('/cancel/<task_id>', methods=['POST'])
def cancel_task(task_id):
    """Cancel a running task"""
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = tasks[task_id]
    if task['status'] not in ['processing', 'queued']:
        return jsonify({'error': 'Task cannot be cancelled'}), 400
    
    # Mark task as cancelled
    task['status'] = 'cancelled'
    task['step'] = 'Cancelled by user'
    
    return jsonify({'status': 'cancelled', 'message': 'Task cancelled successfully'})

@app.route('/download/<task_id>')
def download_file(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = tasks[task_id]
    if task['status'] != 'completed':
        return jsonify({'error': 'Task not completed'}), 400
    
    return send_file(task['output_file'], as_attachment=True, download_name='chord_vocals.mp3')

@app.route('/chords/<task_id>')
def get_chords(task_id):
    """Get chord progression data for a completed task"""
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = tasks[task_id]
    if task['status'] != 'completed':
        return jsonify({'error': 'Task not completed'}), 400
    
    # Load chord data from the saved JSON file
    chords_file = os.path.join(UPLOAD_FOLDER, task_id, 'chords.json')
    if not os.path.exists(chords_file):
        return jsonify({'error': 'Chord data not found'}), 404
    
    try:
        with open(chords_file, 'r') as f:
            chords = json.load(f)
        return jsonify({'chords': chords})
    except Exception as e:
        return jsonify({'error': f'Failed to load chord data: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    # GPU detection for health check
    gpu_info = {}
    try:
        import torch
        gpu_info['pytorch_version'] = torch.__version__
        gpu_info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            gpu_info['cuda_device_count'] = torch.cuda.device_count()
            gpu_info['cuda_device_name'] = torch.cuda.get_device_name()
            gpu_info['cuda_memory_allocated_gb'] = round(torch.cuda.memory_allocated() / 1024**3, 2)
        else:
            gpu_info['cuda_device_count'] = 0
            gpu_info['cuda_device_name'] = 'None'
            gpu_info['cuda_memory_allocated_gb'] = 0
    except ImportError:
        gpu_info['pytorch_version'] = 'Not available'
        gpu_info['cuda_available'] = False
        gpu_info['cuda_device_count'] = 0
        gpu_info['cuda_device_name'] = 'None'
        gpu_info['cuda_memory_allocated_gb'] = 0
    
    return jsonify({
        'status': 'healthy',
        'version': VERSION,
        'name': 'ChordiSpeak',
        'gpu': gpu_info
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

@app.route('/debug/<task_id>')
def debug_task(task_id):
    """Debug endpoint to see task details"""
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = tasks[task_id]
    logs = task_logs.get(task_id, [])
    return jsonify({
        'task_id': task_id,
        'status': task.get('status', 'unknown'),
        'step': task.get('step', 'unknown'),
        'progress': task.get('progress', 0),
        'demucs_percentage': task.get('demucs_percentage', 0),
        'error': task.get('error', None),
        'filename': task.get('filename', 'unknown'),
        'output_file': task.get('output_file', None),
        'logs': logs
    })

@app.route('/logs/<task_id>')
def get_task_logs(task_id):
    """Get debug logs for a specific task"""
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    logs = task_logs.get(task_id, [])
    return jsonify({
        'task_id': task_id,
        'logs': logs
    })

@app.route('/gpu-info')
def get_gpu_info():
    """Get detailed GPU information for debugging"""
    gpu_info = {}
    try:
        import torch
        gpu_info['pytorch_version'] = torch.__version__
        gpu_info['cuda_available'] = torch.cuda.is_available()
        gpu_info['cuda_version'] = torch.version.cuda if torch.cuda.is_available() else 'None'
        
        if torch.cuda.is_available():
            gpu_info['cuda_device_count'] = torch.cuda.device_count()
            gpu_info['cuda_current_device'] = torch.cuda.current_device()
            gpu_info['cuda_device_name'] = torch.cuda.get_device_name()
            gpu_info['cuda_memory_allocated_gb'] = round(torch.cuda.memory_allocated() / 1024**3, 2)
            gpu_info['cuda_memory_reserved_gb'] = round(torch.cuda.memory_reserved() / 1024**3, 2)
            gpu_info['cuda_memory_cached_gb'] = round(torch.cuda.memory_reserved() / 1024**3, 2)
            
            # Get GPU properties
            props = torch.cuda.get_device_properties(0)
            gpu_info['gpu_name'] = props.name
            gpu_info['gpu_memory_total_gb'] = round(props.total_memory / 1024**3, 2)
            gpu_info['gpu_memory_free_gb'] = round((props.total_memory - torch.cuda.memory_allocated()) / 1024**3, 2)
        else:
            gpu_info['cuda_device_count'] = 0
            gpu_info['cuda_current_device'] = -1
            gpu_info['cuda_device_name'] = 'None'
            gpu_info['cuda_memory_allocated_gb'] = 0
            gpu_info['cuda_memory_reserved_gb'] = 0
            gpu_info['cuda_memory_cached_gb'] = 0
            gpu_info['gpu_name'] = 'None'
            gpu_info['gpu_memory_total_gb'] = 0
            gpu_info['gpu_memory_free_gb'] = 0
    except ImportError:
        gpu_info['pytorch_version'] = 'Not available'
        gpu_info['cuda_available'] = False
        gpu_info['cuda_version'] = 'None'
        gpu_info['cuda_device_count'] = 0
        gpu_info['cuda_current_device'] = -1
        gpu_info['cuda_device_name'] = 'None'
        gpu_info['cuda_memory_allocated_gb'] = 0
        gpu_info['cuda_memory_reserved_gb'] = 0
        gpu_info['cuda_memory_cached_gb'] = 0
        gpu_info['gpu_name'] = 'None'
        gpu_info['gpu_memory_total_gb'] = 0
        gpu_info['gpu_memory_free_gb'] = 0
    
    return jsonify(gpu_info)

def test_pronunciation_strategies():
    """Test different pronunciation strategies for letter names"""
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
        }
    }
    
    print("Testing pronunciation strategies:")
    for strategy_name, letters in strategies.items():
        print(f"\n{strategy_name}:")
        for letter, pronunciation in letters.items():
            print(f"  {letter} → {pronunciation}")
    
    return strategies

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    # Disable Flask's default request logging to reduce terminal noise
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(debug=False, host='0.0.0.0', port=port)
