# Project Structure - Advanced AI Video Generator with Multi-Scale Long Sentence Matching

This document outlines the complete project architecture for the advanced AI-powered lyric synchronization video generation system featuring multi-scale semantic search, adaptive language detection, and long sentence optimization.

## 📁 Directory Structure

```
generatevideofromaudioimage/
├── advanced_long_sentence_generator.py  # Main advanced application script
├── context/                            # Input/output directory
│   ├── rapper.jpg                     # Input image file
│   ├── Title _ Judul_Black Chains _ Rantai Hi_cmp.mp3  # Input audio file
│   ├── comparison_lyrics.txt          # Reference lyrics file
│   ├── extracted_lyrics.txt           # AI-extracted lyrics (generated)
│   ├── matched_lyrics.txt            # Detailed matching results (generated)
│   └── advanced_long_sentence_video.mp4 # Generated output video
├── README.md                           # Project overview and quick start
├── INSTALLATION.md                     # Detailed installation instructions
├── USAGE.md                            # Comprehensive usage guide
└── PROJECT_STRUCTURE.md                # This file
```

## 🎯 Core Components

### Main Application (`advanced_long_sentence_generator.py`)

The primary script that orchestrates the entire advanced video generation process:

#### Class: `AdvancedLongSentenceLyricSyncGenerator`

**Key Methods:**
- `__init__()`: Initialize the generator with file paths and analysis file paths
- `_get_audio_duration()`: Extract audio duration using librosa
- `_detect_language()`: Detect English vs Indonesian lyrics
- `_extract_text_from_audio()`: AI text extraction using Whisper for word boundary detection
- `_phonetic_similarity()`: Calculate phonetic similarity using Soundex-like algorithm
- `_hierarchical_forced_alignment()`: Forced alignment with hierarchical matching (sentence → phrase → word)
- `_sliding_window_forced_match()`: DTW-like sliding window forced matching with multiple window sizes
- `_exact_word_match()`: Exact word matching for high-confidence matches
- `_calculate_text_similarity()`: Text similarity calculation with fuzzy matching
- `_build_multi_scale_faiss_index()`: Build Faiss index with small, medium, and large chunks (fallback)
- `_multi_scale_semantic_search()`: Multi-scale semantic search with adaptive thresholds (fallback)
- `_calculate_adaptive_duration()`: Calculate optimal duration based on language and vocal characteristics
- `_match_lyrics_advanced()`: Advanced lyric matching with forced alignment as primary method
- `_save_matched_lyrics()`: Save detailed matching results to file
- `_validate_and_fix_advanced_timing()`: Advanced timing validation with language awareness
- `_fallback_timing()`: Enhanced fallback timing for unmatched lyrics
- `_create_smooth_frames()`: Generate animated video frames with color-coded lyrics
- `generate_video()`: Main orchestration method

**Dependencies:**
- `whisper`: Enhanced AI text extraction
- `moviepy`: Video creation and editing
- `PIL/Pillow`: Image processing
- `librosa`: Audio analysis
- `numpy`: Numerical operations
- `faiss`: Multi-scale semantic search engine
- `sentence_transformers`: Text embeddings for semantic search
- `torch`: PyTorch framework

## 🔧 Technical Architecture

### AI Text Extraction & Forced Alignment Pipeline

1. **Whisper Model Loading**
   - Loads "base" model for balanced speed/accuracy
   - Supports CUDA acceleration when available
   - Caches model for multiple uses

2. **Audio Processing for Word Boundaries**
   - Transcribes audio with word-level timestamps (for boundary detection, not exact transcription)
   - Extracts word boundaries with precise timing
   - Handles multiple languages (detected automatically)

3. **Forced Alignment (Primary Method)**
   - **Hierarchical Forced Alignment**: 
     - Sentence-level matching using sliding windows
     - Phrase-level matching within sentences
     - Word-level phonetic matching
   - **Phonetic Similarity**: Uses Soundex-like algorithm to match words that sound similar
   - **Multiple Metrics**: Combines text similarity, phonetic similarity, word overlap, and position bonus
   - **Lower Thresholds**: Uses 0.5 threshold for hierarchical, 0.4 for sliding window (more lenient than semantic search)
   
4. **Sliding Window DTW-like Approach**
   - Multiple window sizes (±3 words) for flexible matching
   - Tracks last matched position for sequential matching
   - Phonetic matching for individual words within windows

5. **Fallback Methods**
   - **Exact Word Match**: High-confidence exact matching
   - **Faiss Semantic Search**: Multi-scale semantic search with small, medium, and large chunks
   - Maps timing from AI extraction to provided lyrics
   - Ensures sequential timing for complete coverage

### Video Generation Pipeline

1. **Image Processing**
   - Resizes input image to 1024x1024 resolution
   - Applies multi-frequency transformations
   - Creates smooth zoom, pan, and rotation effects

2. **Frame Generation**
   - Generates 30 FPS video frames
   - Applies complex mathematical transformations
   - Adds synchronized lyric text with fade effects

3. **Video Composition**
   - Combines frames with original audio
   - Uses H.264 codec for optimal compression
   - Maintains high quality output

## 🎵 Lyric Synchronization System with Multiple Speakers

### Multiple Speaker Detection Strategy

1. **Speaker Change Detection**
   - Analyzes time gaps between segments (>1.5s threshold)
   - Detects tempo changes (>40% difference)
   - Identifies word length variations (>30% difference)
   - Assigns speaker IDs based on audio characteristics

2. **Overlapping Voices Analysis**
   - Detects segments with >30% overlap
   - Groups speakers by overlap patterns
   - Identifies choruses (>2 speakers) and duets (2 speakers)
   - Analyzes song structure based on voice groups

3. **Song Structure Analysis**
   - Groups lyrics into verses, choruses, bridges, outros
   - Assigns appropriate speakers to each group
   - Handles overlapping vs non-overlapping segments differently

### Text Matching Strategy with Multiple Speakers

1. **Semantic Search with Speaker Awareness**
   - Searches extracted text for each lyric group
   - Uses minimum 35% similarity threshold for overlapping voices
   - Prioritizes semantic matches with speaker attribution
   - Handles overlapping voices with flexible timing rules

2. **Intelligent Timing Validation**
   - Different timing rules for overlapping vs non-overlapping segments
   - Allows larger overlap for choruses (up to 80% of previous duration)
   - Allows smaller overlap for mixed segments (up to 50% of previous duration)
   - Normal overlap fixing for non-overlapping segments (shifts start time)

3. **Sequential Timing Across All Speakers**
   - Ensures all lyrics are displayed from start to finish
   - Maintains proper speaker attribution
   - Provides fallback timing for unmatched lyrics
   - Validates timing with overlapping voices awareness

### Animation System

1. **Multi-Frequency Transformations**
   - Zoom: `1.0 + 0.05 * sin(2π * progress)`
   - Pan X: `0.02 * sin(2π * progress * 4)`
   - Pan Y: `0.02 * cos(2π * progress * 3)`
   - Rotation: `1.0 * sin(2π * progress * 2)`

2. **Text Effects**
   - Fade in/out with 0.5s duration
   - Black stroke outline for readability
   - Centered positioning at bottom of frame (120px from bottom)

## 📊 Performance Characteristics

### Hardware Requirements

- **GPU**: CUDA-compatible (RTX 4090 recommended)
- **RAM**: 8GB+ recommended
- **Storage**: 2GB+ free space for models and output

### Processing Times

- **Text Extraction**: ~30-60 seconds for 4-minute audio
- **Frame Generation**: ~2-3 minutes for 4-minute video
- **Video Encoding**: ~1-2 minutes for final output

### Memory Usage

- **Whisper Model**: ~150MB RAM
- **Frame Generation**: ~500MB RAM peak
- **Video Encoding**: ~200MB RAM

## 🔧 Configuration Options

### Whisper Model Selection

```python
# Available models with trade-offs
"tiny"    # 39MB,  ~32x realtime, lowest accuracy
"base"    # 74MB,  ~16x realtime, balanced (recommended)
"small"   # 244MB, ~6x realtime, good accuracy
"medium"  # 769MB, ~2x realtime, high accuracy
"large"   # 1550MB, ~1x realtime, highest accuracy
```

### Video Quality Settings

```python
# Adjustable parameters
fps = 30                    # Frames per second
image_size = (1024, 1024)   # Output resolution
font_size = 40             # Text size
fade_duration = 0.3        # Text fade effect duration
```

### Text Matching Parameters

```python
# Matching sensitivity with multiple speakers
similarity_threshold = 0.35  # Minimum semantic similarity threshold for overlapping voices
lyric_coverage = 0.90       # Percentage of audio duration for lyrics
chunk_size = 10            # Words per text chunk for semantic search (increased for better context)
overlap_size = 3           # Words overlap between chunks
top_k = 7                 # Number of candidates for semantic search
min_duration = 0.2        # Minimum lyric duration for overlapping voices
max_duration = 15.0       # Maximum lyric duration for overlapping voices
```

## 🚀 Deployment Considerations

### Production Setup

1. **Model Caching**
   - Whisper models are cached locally after first download
   - Reduces startup time for subsequent runs

2. **Error Handling**
   - Graceful fallback for missing files
   - Comprehensive error logging
   - Progress tracking for long operations

3. **Resource Management**
   - Efficient memory usage
   - GPU memory optimization
   - Cleanup of temporary files

### Scalability Options

1. **Batch Processing**
   - Can process multiple audio files
   - Parallel processing capabilities
   - Queue-based processing for large volumes

2. **Cloud Deployment**
   - Docker containerization support
   - Kubernetes deployment ready
   - Auto-scaling capabilities

## 📈 Future Enhancements

### Planned Features

1. **Advanced Text Matching**
   - Machine learning-based text alignment
   - Multi-language support improvements
   - Context-aware matching

2. **Animation Enhancements**
   - More complex transformation algorithms
   - Music-responsive animations
   - Custom animation presets

3. **Performance Optimizations**
   - GPU-accelerated frame generation
   - Parallel processing improvements
   - Memory usage optimization

### Integration Possibilities

1. **API Development**
   - RESTful API for web integration
   - Real-time processing capabilities
   - Batch processing endpoints

2. **Cloud Services**
   - AWS/Azure deployment options
   - Serverless function support
   - CDN integration for output delivery

This architecture provides a solid foundation for AI-powered lyric synchronization while maintaining flexibility for future enhancements and scalability requirements.