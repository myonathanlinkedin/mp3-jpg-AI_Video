# AI Video Generator with Faiss Semantic Search

A Python application that combines MP3 audio with static images to create dynamic videos with advanced AI-powered lyric synchronization using Faiss semantic search.

## 🎯 Final Approach

**AI Text Extraction + Faiss Semantic Search**: The system uses OpenAI Whisper to extract text from audio, then employs Faiss-powered semantic search with Sentence Transformers for intelligent lyric matching and synchronization.

### Key Features

- **AI Text Extraction**: Uses OpenAI Whisper to extract text from MP3 audio with word-level timestamps
- **Faiss Semantic Search**: Advanced semantic matching using Facebook's Faiss library
- **Sentence Transformers**: Uses all-MiniLM-L6-v2 model for semantic embeddings
- **Intelligent Matching**: AI matches extracted text with provided lyrics using semantic similarity
- **Text Polishing**: Provided lyrics serve as reference for polishing AI-generated text
- **Perfect Synchronization**: Text displayed matches perfectly with audio timing
- **Smooth Image Animation**: Complex multi-frequency transformations (zoom, pan, rotation)
- **Sequential Timing**: Ensures all lyrics are displayed from start to finish

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (RTX 4090 recommended)
- Windows 10/11

### Installation

```bash
pip install moviepy pillow librosa numpy torch torchvision openai-whisper faiss-cpu sentence-transformers
```

### Usage

```bash
python ai_lyric_sync_generator.py
```

## 📁 Project Structure

```
generatevideofromaudioimage/
├── ai_lyric_sync_generator.py    # Main AI lyric synchronization script
├── context/                      # Input/output files
│   ├── rapper.jpg               # Input image
│   ├── Title _ Judul_Black Chains _ Rantai Hi_cmp.mp3  # Input audio
│   └── ai_lyric_sync_video.mp4  # Generated video
├── README.md                     # This file
├── INSTALLATION.md               # Detailed installation guide
├── USAGE.md                      # Usage examples and documentation
└── PROJECT_STRUCTURE.md          # Complete project architecture
```

## 🎵 How It Works

1. **AI Text Extraction**: OpenAI Whisper analyzes the MP3 file to extract spoken text with word-level timestamps
2. **Faiss Index Building**: Creates semantic embeddings using Sentence Transformers and builds Faiss index
3. **Semantic Search**: For each lyric, performs semantic search using Faiss to find best matches:
   - Generates embeddings for lyrics using Sentence Transformers
   - Searches Faiss index for semantic similarity
   - Finds best matching audio segments with timing
4. **Text Polishing**: Provided lyrics refine AI-extracted text for accuracy
5. **Sequential Timing**: Ensures all lyrics are displayed from start to finish
6. **Video Generation**: Creates smooth animated frames with:
   - Multi-frequency zoom, pan, and rotation
   - Synchronized lyric display with fade effects
   - Professional video output

## 🎨 Features

- **Smart Text Extraction**: AI determines what's actually being said in the audio
- **Intelligent Matching**: AI matches extracted text with provided lyrics
- **Text Polishing**: Provided lyrics ensure accuracy and proper formatting
- **Perfect Synchronization**: Text appears exactly when spoken in audio
- **Complete Coverage**: All lyrics displayed from start to finish
- **Smooth Animation**: Complex image transformations create engaging visual movement

## 🔧 Technical Details

- **Audio Processing**: `openai-whisper` for speech-to-text extraction
- **Semantic Search**: `faiss-cpu` for high-performance similarity search
- **Text Embeddings**: `sentence-transformers` for semantic text embeddings
- **Text Matching**: Faiss-powered semantic similarity matching
- **Image Processing**: `PIL/Pillow` for image manipulation
- **Video Creation**: `moviepy` for video composition
- **AI Integration**: Whisper + Faiss + Sentence Transformers
- **Sequential Timing**: Ensures complete lyric coverage
- **Warning Suppression**: Triton kernel warnings suppressed for Windows compatibility

## 📝 Example Output

The system generates videos with:
- Text extracted directly from audio using AI
- Provided lyrics used to polish and refine extracted text
- Perfect synchronization between spoken words and displayed text
- All lyrics displayed from start to finish
- Smooth image movement synchronized to music
- Professional fade-in/fade-out effects

## 🎯 Use Cases

- Music video creation with accurate lyrics
- Educational content with synchronized text
- Social media content creation
- Presentation materials with audio-visual elements
- Language learning with precise text matching

## 📚 Documentation

- [INSTALLATION.md](INSTALLATION.md) - Detailed installation instructions
- [USAGE.md](USAGE.md) - Comprehensive usage guide and examples
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Complete project architecture

## 🤝 Contributing

This project focuses on AI-powered text extraction and lyric synchronization for video generation. Contributions are welcome for:
- Enhanced text extraction algorithms
- Improved text matching techniques
- Additional animation effects
- Performance optimizations

## 📄 License

This project is open source and available under the Apache License 2.0.