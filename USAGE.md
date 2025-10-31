# Usage Guide - Advanced AI Video Generator with Multi-Scale Long Sentence Matching

This guide explains how to use the advanced AI-powered lyric synchronization system with multi-scale semantic search and adaptive language detection for creating dynamic videos from MP3 audio and static images.

## 🎯 Final Approach Overview

**Forced Alignment + Hierarchical Matching + Phonetic Similarity**: The system uses OpenAI Whisper for word boundary detection, then employs forced alignment with hierarchical matching (sentence → phrase → word) and phonetic similarity algorithms to match known lyrics directly with audio, even when transcription errors occur. This approach is based on state-of-the-art research and significantly improves lyric matching accuracy.

### Core Workflow

1. **AI Text Extraction** → OpenAI Whisper extracts word-level timestamps (for boundary detection, not exact transcription)
2. **Forced Alignment (Primary)** → Hierarchical matching with phonetic similarity:
   - Sentence-level matching using sliding windows
   - Phrase-level matching within sentences
   - Word-level phonetic matching (Soundex-like algorithm)
3. **Sliding Window DTW-like Approach** → Multiple window sizes (±3 words) for flexible matching
4. **Sequential Matching** → Tracks last matched position to maintain proper lyric order
5. **Fallback Methods** → Exact word match and multi-scale semantic search as fallback
6. **Adaptive Language Detection** → Automatically detects English vs Indonesian lyrics
7. **Adaptive Duration Calculation** → Intelligent timing based on language and vocal characteristics
8. **File-Based Processing** → Reads lyrics from external files and saves detailed results
9. **Advanced Timing Validation** → Language-aware overlap resolution
10. **Video Generation** → Create smooth animated video with color-coded lyrics

## 🚀 Basic Usage

### Single Video Generation

```bash
python advanced_long_sentence_generator.py
```

This will:
- Load `context/rapper.jpg` as the base image
- Process `context/Title _ Judul_Black Chains _ Rantai Hi_cmp.mp3` audio
- Extract text using OpenAI Whisper AI with enhanced segmentation
- Build multi-scale Faiss index (small, medium, large chunks)
- Detect language (English vs Indonesian) for each lyric
- Perform multi-scale semantic search for comprehensive matching
- Generate `context/advanced_long_sentence_video.mp4` with synchronized lyrics
- Save detailed analysis files:
  - `context/extracted_lyrics.txt` - AI-extracted text with timing
  - `context/matched_lyrics.txt` - Detailed matching results
  - `context/comparison_lyrics.txt` - Reference lyrics file

## 🎵 How the Advanced AI Text Extraction Works

### 1. Enhanced Whisper Text Extraction

The AI analyzes the audio file with enhanced segmentation for long sentences:

```python
# Load Whisper model with CUDA optimization
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base", device=device)

# Extract text with enhanced segmentation
result = whisper_model.transcribe(
    audio_path, 
    word_timestamps=True,
    verbose=False
)

# Enhanced segmentation for long sentences
combined_segments = []
current_segment = None

for segment in result['segments']:
    if current_segment is None:
        current_segment = {
            'text': segment['text'].strip(),
            'start': segment['start'],
            'end': segment['end'],
            'words': segment.get('words', [])
        }
    else:
        # Combine short segments that are close together
        segment_duration = segment['end'] - segment['start']
        gap = segment['start'] - current_segment['end']
        
        if segment_duration < 2.0 and gap < 1.0:
            current_segment['text'] += " " + segment['text'].strip()
            current_segment['end'] = segment['end']
            current_segment['words'].extend(segment.get('words', []))
        else:
            combined_segments.append(current_segment)
            current_segment = {
                'text': segment['text'].strip(),
                'start': segment['start'],
                'end': segment['end'],
                'words': segment.get('words', [])
            }
```

### 2. Multi-Scale Faiss Semantic Search

The system uses multiple scales for comprehensive matching:

```python
# Build multi-scale Faiss index
def _build_multi_scale_faiss_index(self, extracted_words):
    text_chunks = []
    chunk_timings = []
    
    # Scale 1: Small chunks (3-6 words) for keyword matching
    small_chunk_size = 4
    small_overlap = 2
    
    # Scale 2: Medium chunks (6-12 words) for phrase matching
    medium_chunk_size = 8
    medium_overlap = 3
    
    # Scale 3: Large chunks (12-20 words) for long sentence matching
    large_chunk_size = 16
    large_overlap = 4
    
    # Generate embeddings for all scales
    embeddings = self.sentence_model.encode(text_chunks)
    
    # Create Faiss index
    dimension = embeddings.shape[1]
    self.faiss_index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    self.faiss_index.add(embeddings)
```

### 3. Adaptive Language Detection

Automatically detects English vs Indonesian lyrics:

```python
def _detect_language(self, text):
    # English words
    english_words = ['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'i', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by', 'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can', 'said', 'there', 'each', 'which', 'she', 'do', 'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'time', 'has', 'two', 'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part']
    
    # Indonesian words
    indonesian_words = ['yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'pada', 'adalah', 'ini', 'itu', 'akan', 'telah', 'sudah', 'belum', 'tidak', 'bukan', 'atau', 'juga', 'hanya', 'saja', 'lebih', 'sangat', 'sekali', 'masih', 'tetap', 'selalu', 'pernah', 'mungkin', 'bisa', 'dapat', 'harus', 'perlu', 'ingin', 'mau', 'sedang', 'lagi', 'baru', 'lama', 'besar', 'kecil', 'tinggi', 'rendah', 'baik', 'buruk', 'benar', 'salah', 'mudah', 'sulit', 'cepat', 'lambat', 'banyak', 'sedikit', 'semua', 'setiap', 'beberapa', 'ada']
    
    text_lower = text.lower()
    english_count = sum(1 for word in english_words if word in text_lower)
    indonesian_count = sum(1 for word in indonesian_words if word in text_lower)
    
    if english_count > indonesian_count:
        return 'english'
    elif indonesian_count > english_count:
        return 'indonesian'
    else:
        return 'mixed'
```

### 4. Advanced Matching Algorithm

Uses adaptive thresholds and similarity boosting:

```python
def _multi_scale_semantic_search(self, lyric_text, text_chunks, chunk_timings, top_k=25):
    lyric_language = self._detect_language(lyric_text)
    word_count = len(lyric_text.split())
    
    # Generate embedding and search
    lyric_embedding = self.sentence_model.encode([lyric_text])
    faiss.normalize_L2(lyric_embedding)
    similarities, indices = self.faiss_index.search(lyric_embedding, top_k)
    
    best_matches = []
    for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
        chunk_language = chunk_timings[idx]['language']
        chunk_scale = chunk_timings[idx]['scale']
        
        # Boost similarity based on multiple factors
        adjusted_similarity = similarity
        
        # Language match boost
        if lyric_language == chunk_language:
            adjusted_similarity *= 1.4
        
        # Scale preference for long sentences
        if word_count > 8:  # Long sentence
            if chunk_scale == 'large':
                adjusted_similarity *= 1.3
            elif chunk_scale == 'medium':
                adjusted_similarity *= 1.1
        
        best_matches.append({
            'similarity': float(adjusted_similarity),
            'original_similarity': float(similarity),
            'text': text_chunks[idx],
            'timing': chunk_timings[idx],
            'language_match': lyric_language == chunk_language,
            'scale_match': chunk_scale
        })
    
    return best_matches
```

## 🎨 Advanced Animation Features

### Smooth Image Movement

The system creates complex, multi-frequency animations:

```python
# Multi-frequency transformations
zoom_factor = 1.0 + 0.1 * math.sin(2 * math.pi * progress * 2) + 0.05 * math.sin(2 * math.pi * progress * 7)
pan_x = 0.1 * math.sin(2 * math.pi * progress * 1.5)
pan_y = 0.1 * math.cos(2 * math.pi * progress * 1.2)
rotation = 2 * math.sin(2 * math.pi * progress * 0.8)
```

### Color-Coded Lyric Display

- **English Lyrics**: White text for clear readability
- **Indonesian Lyrics**: Yellow text for visual distinction
- **Mixed Language**: White text for mixed content
- **Adaptive Fade Effects**: Fade duration adjusts based on language and sentence length
- **Stroke Effects**: Black outline for text readability
- **Position Optimization**: Centered at bottom with proper spacing

### Adaptive Fade Effects

```python
# Adaptive fade effect berdasarkan panjang kalimat dan bahasa
word_count = len(current_lyric_text.split())
if current_language == 'indonesian':
    fade_duration = 1.0 if word_count > 12 else 0.8 if word_count > 8 else 0.6
else:
    fade_duration = 0.8 if word_count > 12 else 0.6 if word_count > 8 else 0.4
```

## 📁 File Structure

```
context/
├── rapper.jpg                                    # Input image
├── Title _ Judul_Black Chains _ Rantai Hi_cmp.mp3  # Input audio
├── comparison_lyrics.txt                         # Reference lyrics file
├── extracted_lyrics.txt                          # AI-extracted lyrics (generated)
├── matched_lyrics.txt                            # Detailed matching results (generated)
└── advanced_long_sentence_video.mp4              # Generated video
```

## ⚙️ Configuration Options

### Whisper Model Selection

```python
# Available models (in order of accuracy vs speed)
whisper_model = whisper.load_model("tiny")    # Fastest, least accurate
whisper_model = whisper.load_model("base")    # Balanced (recommended)
whisper_model = whisper.load_model("small")   # More accurate
whisper_model = whisper.load_model("medium") # High accuracy
whisper_model = whisper.load_model("large")  # Highest accuracy, slowest
```

### Multi-Scale Matching Parameters

```python
# Adjustable parameters in the script
similarity_threshold = 0.15  # Lower threshold for long sentences
small_chunk_size = 4         # Words per small chunk
medium_chunk_size = 8        # Words per medium chunk  
large_chunk_size = 16        # Words per large chunk
fade_duration = 0.8          # Base fade effect duration
font_size = 42              # Text size (reduced 30% from 60px)
```

### Video Output Settings

```python
# Video specifications
fps = 30                # Frames per second
image_size = (1024, 1024)  # Output resolution
```

## 🎯 Use Cases

### 1. Music Video Creation
- Create lyric videos with accurate text extraction
- Generate visual content for music streaming
- Produce promotional materials with perfect sync

### 2. Educational Content
- Language learning with precise text matching
- Music education with accurate lyrics
- Presentation materials with audio-visual elements

### 3. Social Media Content
- Instagram/Facebook video posts
- TikTok-style lyric videos
- YouTube content creation

## 🔧 Advanced Usage

### Custom Semantic Search

You can modify the semantic search parameters:

```python
# In ai_lyric_sync_generator.py
def _semantic_search_lyric(self, lyric_text, text_chunks, chunk_timings, top_k=3):
    # Adjust semantic search sensitivity
    similarity_threshold = 0.3  # Lower = more lenient matching
    chunk_size = 5             # Words per chunk for semantic search
    # Perform Faiss semantic search
    similarities, indices = self.faiss_index.search(lyric_embedding, top_k)
```

### Animation Customization

Modify the animation parameters for different visual effects:

```python
# Adjust animation intensity
zoom_factor = 1.0 + 0.05 * math.sin(2 * math.pi * progress)  # Reduce 0.05 for subtler zoom
pan_x = 0.02 * math.sin(2 * math.pi * progress * 4)          # Reduce 0.02 for less pan
rotation = 1.0 * math.sin(2 * math.pi * progress * 2)         # Reduce 1.0 for less rotation
```

## 📊 Performance Optimization

### GPU Acceleration

The system automatically uses CUDA when available:

```python
# CUDA detection and usage
if torch.cuda.is_available():
    print(f"Using CUDA: {torch.cuda.get_device_name()}")
else:
    print("CUDA not available, using CPU")
```

### Memory Management

- Efficient Whisper model loading with caching
- Optimized text matching with progress tracking
- Smart frame generation with memory management

## 🐛 Troubleshooting

### Common Issues

1. **Triton Kernel Warnings**: Automatically suppressed, normal on Windows
2. **Text Extraction Errors**: Check MP3 file format and audio quality
3. **Memory Issues**: Use smaller Whisper model or reduce audio duration
4. **Missing Lyrics**: Ensure all lyrics are displayed with sequential timing
5. **Faiss Import Errors**: Install faiss-cpu or faiss-gpu

### Performance Tips

- Use "base" Whisper model for balanced speed/accuracy
- Reduce image resolution for quicker generation
- Close other applications to free up GPU memory

## 📈 Output Quality

The system generates professional-quality videos with:

- **Accurate Text Extraction**: AI determines what's actually spoken
- **Perfect Synchronization**: Text appears exactly when spoken
- **Complete Coverage**: All lyrics displayed from start to finish
- **Smooth Animation**: Multi-frequency transformations create natural movement
- **High Resolution**: 1024x1024 output with crisp text rendering
- **Professional Audio**: Maintains original audio quality

## 🎵 Example Results

Generated videos feature:
- Text extracted directly from audio using AI
- Provided lyrics used to polish and refine extracted text
- Perfect synchronization between spoken words and displayed text
- All lyrics displayed from start to finish
- Smooth image movement synchronized to music rhythm
- Professional fade effects for text transitions

This approach combines the best of AI automation with human-curated content for optimal results.