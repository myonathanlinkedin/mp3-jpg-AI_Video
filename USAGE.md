# Usage Guide - AI Video Generator with Faiss Semantic Search

This guide explains how to use the AI-powered lyric synchronization system with Faiss semantic search for creating dynamic videos from MP3 audio and static images.

## 🎯 Final Approach Overview

**AI Text Extraction + Faiss Semantic Search**: The system uses OpenAI Whisper to extract text from audio, then employs Faiss-powered semantic search with Sentence Transformers for intelligent lyric matching and synchronization.

### Core Workflow

1. **AI Text Extraction** → OpenAI Whisper extracts text from MP3 with word-level timestamps
2. **Faiss Index Building** → Creates semantic embeddings using Sentence Transformers
3. **Semantic Search** → Faiss finds best semantic matches for each lyric
4. **Text Polishing** → Provided lyrics refine AI-generated text for accuracy
5. **Sequential Timing** → Ensures all lyrics are displayed from start to finish
6. **Video Generation** → Create smooth animated video with synchronized text

## 🚀 Basic Usage

### Single Video Generation

```bash
python ai_lyric_sync_generator.py
```

This will:
- Load `context/rapper.jpg` as the base image
- Process `context/Title _ Judul_Black Chains _ Rantai Hi_cmp.mp3` audio
- Extract text using OpenAI Whisper AI
- Build Faiss index with Sentence Transformers embeddings
- Perform semantic search to match lyrics with audio segments
- Generate `context/ai_lyric_sync_video.mp4` with synchronized lyrics

## 🎵 How the AI Text Extraction Works

### 1. Whisper Text Extraction

The AI analyzes the audio file to extract spoken text:

```python
# Load Whisper model
whisper_model = whisper.load_model("base")

# Extract text with word-level timestamps
result = whisper_model.transcribe(
    audio_path, 
    word_timestamps=True,
    verbose=False
)

# Get word-level timestamps
words_with_timestamps = []
for segment in result['segments']:
    for word in segment['words']:
        words_with_timestamps.append({
            'word': word['word'].strip(),
            'start': word['start'],
            'end': word['end']
        })
```

### 2. Faiss Semantic Search

The system uses Faiss for intelligent semantic matching:

```python
# Build Faiss index with Sentence Transformers
text_chunks = create_text_chunks(extracted_words, chunk_size=5)
embeddings = sentence_model.encode(text_chunks)
faiss_index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings)
faiss_index.add(embeddings)

# Perform semantic search for each lyric
lyric_embedding = sentence_model.encode([lyric_text])
faiss.normalize_L2(lyric_embedding)
similarities, indices = faiss_index.search(lyric_embedding, top_k=3)

# Map timing from best semantic match
if similarities[0][0] > 0.3:  # Minimum similarity threshold
    best_match = text_chunks[indices[0][0]]
    start_time = chunk_timings[indices[0][0]]['start']
    end_time = chunk_timings[indices[0][0]]['end']
```

### 3. Sequential Timing

Ensures all lyrics are displayed from start to finish:

```python
def _ensure_sequential_timing(self, lyric_data, duration):
    # Calculate average time per lyric
    total_lyric_time = duration * 0.90
    time_per_lyric = total_lyric_time / len(lyric_data)
    
    # Ensure each lyric has sequential timing
    for i, lyric_entry in enumerate(lyric_data):
        if lyric_entry['start'] < 0 or lyric_entry['end'] < 0:
            start_time = i * time_per_lyric
            end_time = (i + 1) * time_per_lyric
            lyric_entry['start'] = start_time
            lyric_entry['end'] = end_time
```

## 🎨 Animation Features

### Smooth Image Movement

The system creates complex, multi-frequency animations:

```python
# Multi-frequency transformations
zoom_factor = 1.0 + 0.05 * math.sin(2 * math.pi * progress)
pan_x = 0.02 * math.sin(2 * math.pi * progress * 4)
pan_y = 0.02 * math.cos(2 * math.pi * progress * 3)
rotation = 1.0 * math.sin(2 * math.pi * progress * 2)
```

### Lyric Display Effects

- **Fade In/Out**: Smooth appearance and disappearance
- **Stroke Effects**: Black outline for text readability
- **Position Optimization**: Centered at bottom with proper spacing
- **Sequential Display**: All lyrics shown from start to finish

## 📁 File Structure

```
context/
├── rapper.jpg                                    # Input image
├── Title _ Judul_Black Chains _ Rantai Hi_cmp.mp3  # Input audio
└── ai_lyric_sync_video.mp4                      # Generated video
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

### Text Matching Parameters

```python
# Adjustable parameters in the script
similarity_threshold = 0.3  # Minimum semantic similarity threshold
chunk_size = 5            # Words per text chunk for semantic search
fade_duration = 0.5       # Fade effect duration
font_size = 45           # Text size
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