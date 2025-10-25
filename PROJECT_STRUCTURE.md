# Project Structure

## Overview
Simple MP3 + Image + Lyrics Video Generator - A streamlined project for creating smooth animated videos with synchronized lyrics.

## Directory Structure

```
generatevideofromaudioimage/
├── context/                                 # Input/Output directory
│   ├── rapper.jpg                          # Source image file
│   ├── Title _ Judul_Black Chains _ Rantai Hi_cmp.mp3  # Source audio file
│   ├── simple_video_with_lyrics.mp4        # Basic video output (19MB)
│   └── smooth_video_with_lyrics.mp4        # Smooth video output (42MB)
├── smooth_video_generator.py               # Main application script
├── .gitignore                              # Git ignore configuration
├── PROJECT_STRUCTURE.md                    # This documentation
└── README.md                               # Project overview and usage
```

## Core Components

### Main Script (`smooth_video_generator.py`)
- **SmoothVideoGenerator Class**: Main video generation engine
- **Animation Engine**: Creates smooth moving frames with multiple effects
- **Lyric Synchronization**: Precise timing calculation for text display
- **Video Processing**: Combines frames and audio using MoviePy

### Context Directory (`context/`)
- **Input Files**: Image and audio source files
- **Output Files**: Generated video files
- **File Management**: Organized input/output structure

## Features Implemented

### Animation System
- **Smooth Zoom**: Easing-based zoom in/out effects
- **Pan Movement**: Natural left/right and up/down motion
- **Subtle Rotation**: Maximum 2-degree rotation for realism
- **Multiple Frequencies**: Combined movement patterns for natural motion

### Lyric System
- **Precise Timing**: Frame-accurate synchronization
- **Fade Effects**: Smooth text appearance and disappearance
- **Dynamic Positioning**: Bottom-center text placement
- **Outline Rendering**: Black outline for text visibility

### Video Processing
- **High FPS**: 30 FPS for smooth playback
- **Quality Codecs**: H.264 video and AAC audio
- **Memory Efficient**: PIL-based frame generation
- **Progress Tracking**: Real-time processing updates

## Technical Specifications

### Dependencies
- **PIL/Pillow**: Image processing and manipulation
- **MoviePy**: Video creation and audio integration
- **librosa**: Audio duration calculation
- **numpy**: Mathematical operations for animations

### Performance
- **Processing Time**: ~4 minutes for 232-second video
- **Memory Usage**: Efficient PIL-based processing
- **Output Quality**: High-definition smooth video
- **File Sizes**: 19MB (basic) / 42MB (smooth)

## Usage Workflow

1. **Setup**: Install dependencies and prepare input files
2. **Configuration**: Edit lyrics and animation parameters
3. **Execution**: Run main script to generate video
4. **Output**: Review generated video in context directory

## Customization Options

### Animation Parameters
- Zoom factor and frequency
- Pan movement range and speed
- Rotation angle limits
- Frame rate and quality settings

### Text Customization
- Font size and style
- Text position and color
- Outline thickness and color
- Fade timing and effects

### Video Settings
- Output resolution and quality
- Audio codec and bitrate
- Video codec and compression
- Frame rate and duration

## File Management

### Input Requirements
- **Image**: JPG/PNG format, any resolution
- **Audio**: MP3 format, any duration
- **Lyrics**: Text array in script

### Output Specifications
- **Format**: MP4 container
- **Video**: H.264 codec, 30 FPS
- **Audio**: AAC codec, original quality
- **Resolution**: Matches input image

## Maintenance

### Regular Tasks
- Update dependencies as needed
- Clean up temporary files
- Monitor output file sizes
- Test with different input formats

### Future Enhancements
- GUI interface for easier use
- Batch processing capabilities
- More animation effects
- Advanced text styling options