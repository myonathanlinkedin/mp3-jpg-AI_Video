# Advanced AI Video Generator with Multi-Scale Long Sentence Matching

A cutting-edge AI-powered video generator that creates perfectly synchronized lyric videos from static images and audio files. This project uses advanced AI techniques including Whisper for speech-to-text, multi-scale Faiss semantic search, and adaptive language detection to achieve superior lyric synchronization, especially for long sentences.

## 🎯 Final Approach

**Multi-Scale AI Text Extraction + Advanced Semantic Matching + Long Sentence Optimization**: The system uses OpenAI Whisper with enhanced segmentation to extract text from audio, then employs multi-scale Faiss-powered semantic search with adaptive language detection for intelligent lyric matching, especially optimized for long sentences that were previously missed.

### Key Features

- **Enhanced AI Text Extraction**: Uses OpenAI Whisper with enhanced segmentation for better long sentence capture
- **Multi-Scale Semantic Search**: Uses Faiss with small, medium, and large chunks to capture both short phrases and long sentences
- **Adaptive Language Detection**: Automatically detects English vs Indonesian lyrics for optimal timing
- **Long Sentence Optimization**: Specialized algorithms for handling complex, long sentences
- **File-Based Lyrics**: Reads lyrics from external files instead of hardcoded arrays
- **Advanced Timing Validation**: Intelligent timing adjustment based on language, word count, and vocal speed
- **Color-Coded Display**: Different colors for English (white) and Indonesian (yellow) lyrics
- **Adaptive Fade Effects**: Fade duration adjusts based on language and sentence length
- **Perfect Synchronization**: Text displayed matches perfectly with audio timing
- **Smooth Image Animation**: Complex multi-frequency transformations (zoom, pan, rotation)
- **Complete Coverage**: All lyrics displayed from start to finish with proper timing

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
python advanced_long_sentence_generator.py
```

## 📁 Project Structure

```
generatevideofromaudioimage/
├── advanced_long_sentence_generator.py  # Main advanced AI lyric synchronization script
├── context/                              # Input/output files
│   ├── rapper.jpg                       # Input image
│   ├── Title _ Judul_Black Chains _ Rantai Hi_cmp.mp3  # Input audio
│   ├── comparison_lyrics.txt             # Comparison lyrics file
│   ├── extracted_lyrics.txt              # AI-extracted lyrics (generated)
│   ├── matched_lyrics.txt               # Matching results (generated)
│   └── advanced_long_sentence_video.mp4 # Generated video
├── README.md                             # This file
├── INSTALLATION.md                       # Detailed installation guide
├── USAGE.md                              # Usage examples and documentation
└── PROJECT_STRUCTURE.md                  # Complete project architecture
```

## 🎵 How It Works

1. **Enhanced AI Text Extraction**: OpenAI Whisper analyzes the MP3 file with enhanced segmentation to extract spoken text with word-level timestamps, optimized for long sentences
2. **Multi-Scale Index Building**: Creates semantic embeddings using Sentence Transformers and builds Faiss index with small, medium, and large chunks
3. **Adaptive Language Detection**: Automatically detects English vs Indonesian lyrics for optimal timing and display
4. **Multi-Scale Semantic Search**: For each lyric, performs semantic search using multiple scales:
   - Small chunks (3-6 words) for keyword matching
   - Medium chunks (6-12 words) for phrase matching  
   - Large chunks (12-20 words) for long sentence matching
5. **Advanced Matching Algorithm**: Uses adaptive thresholds and similarity boosting based on:
   - Language match (English/Indonesian)
   - Sentence length and scale preference
   - Word count similarity
6. **Adaptive Duration Calculation**: Calculates optimal lyric duration based on:
   - Language (Indonesian typically longer)
   - Word count (longer sentences get more time)
   - Vocal speed analysis
7. **File-Based Processing**: Reads lyrics from `comparison_lyrics.txt` and saves results to separate files
8. **Advanced Timing Validation**: Intelligent timing adjustment with language-aware overlap resolution
9. **Video Generation**: Creates smooth animated frames with:
   - Multi-frequency zoom, pan, and rotation
   - Color-coded lyric display (white for English, yellow for Indonesian)
   - Adaptive fade effects based on language and sentence length
   - Professional video output

## 🎨 Features

- **Enhanced Text Extraction**: AI determines what's actually being said in the audio with optimized segmentation
- **Multi-Scale Matching**: Captures both short phrases and long sentences using multiple chunk sizes
- **Adaptive Language Detection**: Automatically identifies English vs Indonesian lyrics
- **Long Sentence Optimization**: Specialized algorithms for handling complex, long sentences
- **File-Based Processing**: Reads lyrics from external files and saves detailed results
- **Intelligent Matching**: AI matches extracted text with provided lyrics using advanced semantic search
- **Adaptive Timing**: Timing adjusts based on language, word count, and vocal speed
- **Perfect Synchronization**: Text appears exactly when spoken in audio
- **Complete Coverage**: All lyrics displayed from start to finish with proper timing
- **Color-Coded Display**: Visual distinction between English and Indonesian lyrics
- **Smooth Animation**: Complex image transformations create engaging visual movement

## 🔧 Technical Details

- **Audio Processing**: `openai-whisper` for enhanced speech-to-text extraction with optimized segmentation
- **Multi-Scale Semantic Search**: `faiss-cpu` for high-performance similarity search with multiple chunk sizes
- **Text Embeddings**: `sentence-transformers` for semantic text embeddings
- **Language Detection**: Custom algorithm for English vs Indonesian lyric detection
- **Adaptive Timing**: Intelligent duration calculation based on language and vocal characteristics
- **Text Matching**: Multi-scale Faiss-powered semantic similarity matching
- **Image Processing**: `PIL/Pillow` for image manipulation
- **Video Creation**: `moviepy` for video composition
- **AI Integration**: Whisper + Multi-Scale Faiss + Sentence Transformers + Language Detection
- **File Management**: Automatic saving of extracted lyrics, comparison lyrics, and matching results
- **Warning Suppression**: Triton kernel warnings suppressed for Windows compatibility

## 📝 Example Output

The system generates videos with:
- Text extracted directly from audio using AI with enhanced segmentation for long sentences
- Multi-scale semantic matching for both short phrases and long sentences
- Adaptive language detection for optimal timing (English vs Indonesian)
- Color-coded lyric display (white for English, yellow for Indonesian)
- Perfect synchronization between spoken words and displayed text
- All lyrics displayed from start to finish with proper timing
- Adaptive fade effects based on language and sentence length
- Smooth image movement synchronized to music
- Professional fade-in/fade-out effects
- Detailed analysis files saved for debugging and analysis

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

This project focuses on advanced AI-powered text extraction and multi-scale lyric synchronization for video generation. Contributions are welcome for:
- Enhanced multi-scale text extraction algorithms
- Improved long sentence matching techniques
- Additional language detection capabilities
- Performance optimizations for large-scale processing
- Advanced animation effects

## 📄 License

This project is open source and available under the Apache License 2.0.