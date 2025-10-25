# Simple MP3 + Image + Lyrics Video Generator

A simple Python script to create smooth animated videos by combining MP3 audio with moving images and synchronized lyrics.

## Features

- **Smooth Image Animation**: Multiple frequency movements with zoom, pan, and subtle rotation
- **Synchronized Lyrics**: Text appears and fades with precise timing
- **High Quality Output**: 30 FPS video with AAC audio
- **Easy to Use**: Simple script with minimal dependencies

## Requirements

- Python 3.8+
- PIL/Pillow
- MoviePy
- librosa
- numpy

## Installation

```bash
pip install moviepy pillow librosa numpy
```

## Usage

1. Place your image and MP3 file in the `context/` folder
2. Edit the lyrics in `smooth_video_generator.py`
3. Run the script:

```bash
python smooth_video_generator.py
```

## File Structure

```
generatevideofromaudioimage/
├── context/
│   ├── rapper.jpg                           # Input image
│   ├── Title _ Judul_Black Chains _ Rantai Hi_cmp.mp3  # Input audio
│   ├── simple_video_with_lyrics.mp4         # Basic output
│   └── smooth_video_with_lyrics.mp4         # Smooth output
├── smooth_video_generator.py                # Main script
├── .gitignore                               # Git ignore file
├── PROJECT_STRUCTURE.md                     # Project documentation
└── README.md                                # This file
```

## How It Works

1. **Image Processing**: Creates smooth moving frames with multiple animation effects
2. **Lyric Synchronization**: Calculates precise timing for each lyric line
3. **Video Generation**: Combines frames with audio using MoviePy
4. **Output**: Creates high-quality MP4 video with synchronized lyrics

## Animation Effects

- **Zoom**: Smooth zoom in/out with easing
- **Pan**: Natural left/right and up/down movement
- **Rotation**: Subtle rotation (max 2 degrees)
- **Multiple Frequencies**: Combines different movement patterns for natural motion

## Customization

- Edit `lyrics` list in the script to change text
- Modify animation parameters in `_create_smooth_frames_with_lyrics()`
- Adjust font size, position, and colors for text
- Change FPS and video quality settings

## Output

The script generates a smooth MP4 video with:
- Moving background image
- Synchronized lyrics at the bottom
- High-quality audio integration
- Professional-looking transitions

## License

This project is open source and available under the MIT License.