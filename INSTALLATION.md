# Installation Guide - Advanced AI Video Generator with Multi-Scale Long Sentence Matching

This guide provides detailed installation instructions for the advanced AI-powered lyric synchronization video generation system featuring multi-scale semantic search, adaptive language detection, and long sentence optimization.

## 🎯 System Requirements

### Hardware Requirements

- **GPU**: CUDA-compatible GPU (RTX 4090 recommended)
- **RAM**: 8GB+ recommended (16GB+ for optimal performance)
- **Storage**: 2GB+ free space for models and output
- **CPU**: Multi-core processor (4+ cores recommended)

### Software Requirements

- **Operating System**: Windows 10/11, macOS 10.15+, or Linux Ubuntu 18.04+
- **Python**: 3.8 or higher
- **CUDA**: 11.8+ (for GPU acceleration)

## 🚀 Quick Installation

### Step 1: Install Python Dependencies

```bash
pip install moviepy pillow librosa numpy torch torchvision openai-whisper faiss-cpu sentence-transformers
```

### Step 2: Verify Installation

```bash
python -c "import whisper; print('Whisper installed successfully')"
python -c "import moviepy; print('MoviePy installed successfully')"
python -c "import librosa; print('Librosa installed successfully')"
```

## 🔧 Detailed Installation

### Python Environment Setup

#### Option 1: Using pip (Recommended)

```bash
# Create virtual environment
python -m venv ai_video_env

# Activate virtual environment
# Windows:
ai_video_env\Scripts\activate
# macOS/Linux:
source ai_video_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install moviepy pillow librosa numpy torch torchvision openai-whisper faiss-cpu sentence-transformers
```

#### Option 2: Using conda

```bash
# Create conda environment
conda create -n ai_video_env python=3.9

# Activate environment
conda activate ai_video_env

# Install dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install moviepy pillow librosa numpy openai-whisper faiss-cpu sentence-transformers
```

### CUDA Installation (for GPU acceleration)

#### Windows

1. **Download CUDA Toolkit**
   - Visit: https://developer.nvidia.com/cuda-downloads
   - Download CUDA Toolkit 11.8 or later
   - Run installer with default settings

2. **Verify CUDA Installation**
   ```bash
   nvcc --version
   ```

3. **Install PyTorch with CUDA**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

#### macOS

```bash
# Install CUDA via Homebrew
brew install cuda

# Verify installation
nvcc --version
```

#### Linux (Ubuntu)

```bash
# Install CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repository-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repository-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repository-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Verify installation
nvcc --version
```

### Dependency Details

#### Core Dependencies

- **moviepy**: Video creation and editing
- **pillow**: Image processing
- **librosa**: Audio analysis and processing
- **numpy**: Numerical operations

#### AI/ML Dependencies

- **openai-whisper**: Speech-to-text AI model
- **torch**: PyTorch deep learning framework
- **torchvision**: Computer vision utilities
- **faiss-cpu**: Semantic search engine
- **sentence-transformers**: Text embeddings for semantic search

## 🔍 Verification Steps

### 1. Check Python Installation

```bash
python --version
# Should show Python 3.8 or higher
```

### 2. Check CUDA Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 3. Check Whisper Installation

```bash
python -c "import whisper; model = whisper.load_model('base'); print('Whisper model loaded successfully')"
```

### 4. Check All Dependencies

```bash
python -c "
import moviepy, PIL, librosa, numpy, whisper, torch, faiss, sentence_transformers
print('All dependencies installed successfully!')
"
```

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Not Available

**Problem**: `CUDA available: False`

**Solutions**:
- Verify CUDA installation: `nvcc --version`
- Reinstall PyTorch with CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- Check GPU compatibility

#### 2. Whisper Installation Issues

**Problem**: `ModuleNotFoundError: No module named 'whisper'`

**Solutions**:
```bash
pip uninstall openai-whisper
pip install openai-whisper
```

#### 3. Triton Kernel Warnings

**Problem**: Triton kernel warnings on Windows

**Solutions**:
- Warnings are automatically suppressed in the script
- Normal behavior on Windows systems
- Does not affect functionality

#### 4. Faiss Installation Issues

**Problem**: `ModuleNotFoundError: No module named 'faiss'`

**Solutions**:
```bash
pip install faiss-cpu
# For GPU version: pip install faiss-gpu
```

#### 5. Memory Issues

**Problem**: Out of memory errors

**Solutions**:
- Use smaller Whisper model: `whisper.load_model("tiny")`
- Reduce image resolution in script
- Close other applications

### Performance Optimization

#### 1. GPU Memory Optimization

```python
# In script, add memory management
import torch
torch.cuda.empty_cache()  # Clear GPU memory
```

#### 2. Model Caching

```python
# Models are automatically cached after first download
# Cache location: ~/.cache/whisper/
```

#### 3. Batch Processing

```python
# For multiple files, reuse model
whisper_model = whisper.load_model("base")
# Use same model for all files
```

## 📊 System Performance

### Expected Performance (RTX 4090)

- **Text Extraction**: ~30-60 seconds for 4-minute audio
- **Frame Generation**: ~2-3 minutes for 4-minute video
- **Video Encoding**: ~1-2 minutes for final output
- **Total Time**: ~4-6 minutes for complete video

### Memory Usage

- **Whisper Model**: ~150MB RAM
- **Frame Generation**: ~500MB RAM peak
- **Video Encoding**: ~200MB RAM
- **Total Peak**: ~850MB RAM

## 🔄 Updates and Maintenance

### Updating Dependencies

```bash
# Update all packages
pip install --upgrade moviepy pillow librosa numpy torch torchvision openai-whisper faiss-cpu sentence-transformers

# Update specific packages
pip install --upgrade openai-whisper
pip install --upgrade torch torchvision torchaudio
```

### Cleaning Up

```bash
# Clear pip cache
pip cache purge

# Clear Whisper model cache
rm -rf ~/.cache/whisper/

# Clear PyTorch cache
rm -rf ~/.cache/torch/
```

## 📚 Additional Resources

### Documentation Links

- [Whisper Documentation](https://github.com/openai/whisper)
- [MoviePy Documentation](https://zulko.github.io/moviepy/)
- [Librosa Documentation](https://librosa.org/doc/latest/)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Community Support

- [GitHub Issues](https://github.com/openai/whisper/issues)
- [PyTorch Forums](https://discuss.pytorch.org/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/whisper)

This installation guide ensures you have all the necessary components to run the AI lyric synchronization video generator successfully.