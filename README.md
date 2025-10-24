# AI Video Generator

A dynamic AI video generator that automatically detects hardware capabilities and adapts model selection and processing parameters accordingly.

## Features

- **Dynamic Hardware Detection**: Automatically detects GPU, CPU, RAM capabilities
- **Adaptive Model Selection**: Chooses optimal AI model based on hardware tier
- **True AI Animation**: Generates natural video movement from static images
- **Multi-tier Support**: Ultra, High, Medium, Low hardware tiers
- **Local Processing**: No cloud dependency for true AI animation
- **Ken Burns Fallback**: Traditional animation effects when AI models unavailable

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic example
python examples/basic_usage.py
```

## Hardware Tiers

- **Ultra**: RTX 4090 Mobile (16GB VRAM) - All models, maximum quality
- **High**: RTX 4080/3080 (10-16GB VRAM) - High quality models
- **Medium**: RTX 3060/4060 (8-12GB VRAM) - Medium quality models  
- **Low**: Integrated Graphics/CPU - Basic animation effects

## Supported Models

- **Stable Video Diffusion XL**: Highest quality AI animation (requires authentication)
- **Stable Video Diffusion**: High quality AI animation (requires authentication)
- **AnimateDiff**: Creative AI animation (requires authentication)
- **Ken Burns Effect**: Traditional animation fallback (always available)

## Project Structure

```
ai_video_generator/
├── README.md
├── requirements.txt
├── PROJECT_STRUCTURE.md
├── core/
│   ├── __init__.py
│   ├── hardware_detector.py      # GPU/CPU/RAM detection
│   ├── model_manager.py          # Dynamic model loading
│   ├── video_generator.py        # Main video generation
│   ├── audio_processor.py        # Audio processing utilities
│   ├── base_model.py            # Base model interface
│   └── ken_burns.py             # Ken Burns effect implementation
├── examples/
│   ├── __init__.py
│   └── basic_usage.py           # Basic usage example
└── context/
    ├── rapper.jpg               # Input image
    └── Title _ Judul_Black Chains _ Rantai Hi_cmp.mp3  # Input audio
```

## Usage

```python
from core.video_generator import VideoGenerator

# Auto-detect hardware and initialize
generator = VideoGenerator()

# Generate video
output_path = generator.generate_video(
    image_path="context/rapper.jpg",
    audio_path="context/Title _ Judul_Black Chains _ Rantai Hi_cmp.mp3",
    output_path="context/generated_video.mp4"
)
```

## License

MIT License
