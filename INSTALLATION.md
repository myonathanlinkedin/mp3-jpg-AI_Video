# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Windows/Linux/macOS operating system

## Quick Installation

### 1. Install Python Dependencies

```bash
pip install moviepy pillow librosa numpy
```

### 2. Verify Installation

```bash
python -c "import moviepy, PIL, librosa, numpy; print('All dependencies installed successfully!')"
```

## Detailed Installation

### Required Packages

| Package | Version | Purpose |
|---------|---------|---------|
| moviepy | Latest | Video creation and editing |
| pillow | Latest | Image processing |
| librosa | Latest | Audio analysis |
| numpy | Latest | Mathematical operations |

### Installation Commands

```bash
# Core video processing
pip install moviepy

# Image manipulation
pip install pillow

# Audio processing
pip install librosa

# Numerical computing
pip install numpy
```

## System Requirements

### Minimum Requirements
- **RAM**: 4GB
- **Storage**: 2GB free space
- **CPU**: Dual-core processor
- **GPU**: Not required (CPU-based processing)

### Recommended Requirements
- **RAM**: 8GB or more
- **Storage**: 5GB free space
- **CPU**: Quad-core processor or better
- **GPU**: Optional (for faster processing)

## Troubleshooting

### Common Issues

#### 1. MoviePy Installation Issues
```bash
# If MoviePy fails to install
pip install --upgrade pip
pip install moviepy --no-cache-dir
```

#### 2. PIL/Pillow Conflicts
```bash
# Uninstall old PIL if present
pip uninstall PIL
pip install pillow
```

#### 3. Audio Codec Issues
```bash
# Install additional audio support
pip install imageio-ffmpeg
```

### Platform-Specific Notes

#### Windows
- Ensure Visual C++ Redistributable is installed
- Use Command Prompt or PowerShell as Administrator if needed

#### Linux
- Install system audio libraries: `sudo apt-get install libsndfile1`
- May need additional codec support

#### macOS
- Use Homebrew for system dependencies: `brew install ffmpeg`
- Ensure Xcode command line tools are installed

## Verification

### Test Installation
Run this test script to verify everything works:

```python
import moviepy.editor as mp
from PIL import Image
import librosa
import numpy as np

print("✅ All dependencies installed successfully!")
print("Ready to create smooth videos with lyrics!")
```

### Performance Test
Create a small test video to verify performance:

```python
from smooth_video_generator import SmoothVideoGenerator

# Test with small files
generator = SmoothVideoGenerator()
# Run with test files to verify functionality
```

## Next Steps

After successful installation:

1. **Prepare Input Files**: Place image and MP3 in `context/` folder
2. **Edit Lyrics**: Modify lyrics in `smooth_video_generator.py`
3. **Run Script**: Execute `python smooth_video_generator.py`
4. **Check Output**: Review generated video in `context/` folder

## Support

If you encounter issues:

1. Check Python version: `python --version`
2. Verify pip installation: `pip --version`
3. Update all packages: `pip install --upgrade -r requirements.txt`
4. Check system requirements and dependencies
