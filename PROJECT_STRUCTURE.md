# AI Video Generator - Project Structure

## Overview
Dynamic AI video generator that automatically detects hardware capabilities (GPU, CPU, RAM) and adapts model selection and processing parameters accordingly.

## Project Structure

```
ai_video_generator/
├── README.md
├── requirements.txt
├── config/
│   ├── __init__.py
│   ├── hardware_config.py      # Hardware detection & configuration
│   ├── model_config.py         # Model selection & parameters
│   └── settings.py            # Global settings
├── core/
│   ├── __init__.py
│   ├── hardware_detector.py    # GPU/CPU/RAM detection
│   ├── model_manager.py        # Dynamic model loading
│   ├── video_generator.py      # Main video generation logic
│   └── audio_processor.py     # Audio processing utilities
├── models/
│   ├── __init__.py
│   ├── stable_video_diffusion.py
│   ├── animate_diff.py
│   ├── ken_burns.py           # Fallback for low-end hardware
│   └── base_model.py          # Base model interface
├── utils/
│   ├── __init__.py
│   ├── file_utils.py          # File handling utilities
│   ├── video_utils.py         # Video processing utilities
│   ├── memory_utils.py        # Memory management
│   └── progress_tracker.py    # Progress tracking
├── gui/
│   ├── __init__.py
│   ├── main_window.py         # Main GUI window
│   ├── hardware_info.py       # Hardware info display
│   └── progress_dialog.py    # Progress dialog
├── tests/
│   ├── __init__.py
│   ├── test_hardware_detector.py
│   ├── test_model_manager.py
│   └── test_video_generator.py
├── examples/
│   ├── basic_usage.py
│   ├── batch_processing.py
│   └── custom_prompts.py
├── scripts/
│   ├── install_models.py      # Model installation script
│   ├── benchmark.py          # Hardware benchmarking
│   └── cleanup.py            # Cleanup utilities
└── docs/
    ├── installation.md
    ├── usage.md
    ├── hardware_requirements.md
    └── troubleshooting.md
```

## Core Components

### 1. Hardware Detection (`core/hardware_detector.py`)
```python
class HardwareDetector:
    def detect_gpu(self) -> dict
    def detect_cpu(self) -> dict
    def detect_ram(self) -> dict
    def get_hardware_tier(self) -> str  # "low", "medium", "high", "ultra"
    def estimate_capabilities(self) -> dict
```

### 2. Model Manager (`core/model_manager.py`)
```python
class ModelManager:
    def select_optimal_model(self, hardware_tier: str) -> str
    def load_model(self, model_name: str) -> BaseModel
    def get_model_requirements(self, model_name: str) -> dict
    def optimize_model_for_hardware(self, model, hardware_info: dict)
```

### 3. Video Generator (`core/video_generator.py`)
```python
class VideoGenerator:
    def __init__(self, hardware_detector: HardwareDetector)
    def generate_video(self, image_path: str, audio_path: str, 
                      output_path: str, **kwargs) -> str
    def generate_batch(self, inputs: list) -> list
    def get_estimated_time(self, duration: float) -> float
```

## Hardware Tiers

### Ultra Tier (RTX 4090 Mobile - 16GB VRAM)
- **Models**: Stable Video Diffusion XL, RunwayML Gen-2
- **Quality**: Maximum (4K, 60fps)
- **Processing**: 2-5 minutes for 10 seconds
- **Features**: All advanced features enabled

### High Tier (RTX 4080/3080 - 10-16GB VRAM)
- **Models**: Stable Video Diffusion, AnimateDiff
- **Quality**: High (1080p, 30fps)
- **Processing**: 3-7 minutes for 10 seconds
- **Features**: Most features enabled

### Medium Tier (RTX 3060/4060 - 8-12GB VRAM)
- **Models**: AnimateDiff, Lightweight SVD
- **Quality**: Medium (720p, 24fps)
- **Processing**: 5-10 minutes for 10 seconds
- **Features**: Basic features enabled

### Low Tier (Integrated Graphics/CPU)
- **Models**: Ken Burns Effect, Basic Animation
- **Quality**: Basic (480p, 15fps)
- **Processing**: 10-30 minutes for 10 seconds
- **Features**: Limited features

## Dynamic Configuration

### Auto-Detection Features
- **GPU Detection**: CUDA availability, VRAM size, compute capability
- **CPU Detection**: Core count, clock speed, architecture
- **RAM Detection**: Total memory, available memory
- **Storage Detection**: Available disk space for models

### Adaptive Parameters
- **Batch Size**: Automatically adjusted based on VRAM
- **Model Precision**: FP16/FP32 based on hardware support
- **Processing Chunks**: Memory-aware chunking
- **Quality Settings**: Auto-scaling based on capabilities

## Installation Requirements

### Minimum Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 8GB RAM
- 50GB free disk space

### Recommended Requirements
- Python 3.10+
- CUDA 12.0+
- RTX 4090 Mobile (16GB VRAM)
- 32GB RAM
- 100GB free disk space

## Usage Examples

### Basic Usage
```python
from ai_video_generator import VideoGenerator

# Auto-detect hardware and initialize
generator = VideoGenerator()

# Generate video
output_path = generator.generate_video(
    image_path="input.jpg",
    audio_path="input.mp3",
    output_path="output.mp4"
)
```

### Advanced Usage
```python
# Custom configuration
generator = VideoGenerator(
    hardware_tier="ultra",
    model_preference="stable_video_diffusion",
    quality="maximum"
)

# Batch processing
inputs = [
    {"image": "img1.jpg", "audio": "audio1.mp3"},
    {"image": "img2.jpg", "audio": "audio2.mp3"}
]

results = generator.generate_batch(inputs)
```

## Performance Optimization

### Memory Management
- **Dynamic VRAM allocation** based on available memory
- **Model caching** for faster subsequent generations
- **Garbage collection** optimization
- **Memory monitoring** and cleanup

### Processing Optimization
- **Multi-threading** for I/O operations
- **GPU utilization** monitoring
- **Batch processing** for efficiency
- **Progress tracking** and estimation

## Error Handling

### Hardware Errors
- **Insufficient VRAM**: Automatic model downgrade
- **CUDA errors**: Fallback to CPU processing
- **Memory overflow**: Chunk-based processing
- **Model loading failures**: Alternative model selection

### Processing Errors
- **File format errors**: Automatic conversion
- **Audio sync issues**: Duration validation
- **Quality degradation**: Parameter adjustment
- **Timeout errors**: Retry with reduced quality

## Monitoring & Logging

### Hardware Monitoring
- **Real-time GPU usage** tracking
- **Memory consumption** monitoring
- **Temperature** monitoring (if available)
- **Performance metrics** collection

### Processing Logging
- **Generation progress** tracking
- **Error logging** and reporting
- **Performance statistics** collection
- **User feedback** integration

## Future Enhancements

### Planned Features
- **Real-time preview** during generation
- **Custom model training** support
- **API integration** for cloud services
- **Advanced effects** and filters
- **Multi-GPU support** for scaling

### Research Areas
- **Model optimization** for mobile GPUs
- **Memory-efficient** processing techniques
- **Quality-speed** trade-off optimization
- **Hardware-specific** model variants
