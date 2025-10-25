# Usage Guide

## Quick Start

### 1. Prepare Your Files
Place your image and MP3 file in the `context/` folder:
```
context/
├── your_image.jpg
└── your_audio.mp3
```

### 2. Edit Lyrics
Open `smooth_video_generator.py` and modify the lyrics array:
```python
lyrics = [
    "Your first lyric line",
    "Your second lyric line",
    "Your third lyric line",
    # Add more lines as needed
]
```

### 3. Run the Script
```bash
python smooth_video_generator.py
```

### 4. Check Output
Your video will be saved as `context/smooth_video_with_lyrics.mp4`

## Detailed Usage

### File Preparation

#### Image Requirements
- **Format**: JPG, PNG, or other PIL-supported formats
- **Resolution**: Any resolution (will be processed as-is)
- **Size**: Larger images may take longer to process
- **Content**: Works best with clear, high-contrast images

#### Audio Requirements
- **Format**: MP3, WAV, or other MoviePy-supported formats
- **Duration**: Any length (longer videos take more time)
- **Quality**: Higher quality audio produces better results
- **Content**: Any audio content (music, speech, etc.)

### Lyrics Configuration

#### Basic Lyrics Setup
```python
lyrics = [
    "Black Chains",
    "Rantai Hitam", 
    "Mengikat jiwa",
    "Dalam kegelapan",
    "Mencari cahaya",
    "Di tengah malam"
]
```

#### Advanced Lyrics Configuration
```python
# For longer songs, add more lines
lyrics = [
    "Verse 1 - Line 1",
    "Verse 1 - Line 2", 
    "Chorus - Line 1",
    "Chorus - Line 2",
    "Verse 2 - Line 1",
    "Verse 2 - Line 2",
    # Continue for full song
]
```

### Customization Options

#### Animation Settings
Modify these parameters in the script for different effects:

```python
# Zoom settings
zoom_base = 1.0                    # Base zoom level
zoom_variation = 0.3              # Zoom variation amount

# Pan settings  
pan_x = int(80 * math.sin(...))    # Horizontal movement range
pan_y = int(60 * math.cos(...))    # Vertical movement range

# Rotation settings
rotation_angle = 2 * math.sin(...) # Maximum rotation (degrees)
```

#### Text Settings
Customize text appearance:

```python
# Font settings
font = ImageFont.truetype("arial.ttf", 60)  # Font size

# Text position
text_y = height - text_height - 80           # Distance from bottom

# Text colors
text_color = (255, 255, 255)                 # White text
outline_color = (0, 0, 0)                    # Black outline
```

#### Video Settings
Adjust output quality:

```python
# Frame rate
fps = 30                                      # Higher = smoother, larger file

# Video codec
codec = 'libx264'                            # H.264 compression
audio_codec = 'aac'                          # AAC audio
```

## Advanced Usage

### Batch Processing
Process multiple videos by modifying the script:

```python
# List of input files
input_files = [
    ("image1.jpg", "audio1.mp3", "output1.mp4"),
    ("image2.jpg", "audio2.mp3", "output2.mp4"),
    ("image3.jpg", "audio3.mp3", "output3.mp4")
]

# Process each file
for image_path, audio_path, output_path in input_files:
    generator.create_smooth_animation(image_path, audio_path, lyrics, output_path)
```

### Custom Animation Patterns
Create different movement patterns:

```python
# Slow, gentle movement
pan_x = int(30 * math.sin(progress * math.pi * 1.5))
pan_y = int(20 * math.cos(progress * math.pi * 1.2))

# Fast, dynamic movement  
pan_x = int(120 * math.sin(progress * math.pi * 4))
pan_y = int(80 * math.cos(progress * math.pi * 3))
```

### Timing Adjustments
Fine-tune lyric timing:

```python
# Faster lyric changes
total_lyric_time = duration * 0.6  # 60% of duration for lyrics

# Slower lyric changes
total_lyric_time = duration * 0.9  # 90% of duration for lyrics
```

## Output Options

### File Formats
The script generates MP4 files with:
- **Video**: H.264 codec, 30 FPS
- **Audio**: AAC codec, original quality
- **Container**: MP4 format

### Quality Settings
- **High Quality**: Default settings (42MB for 232s video)
- **Medium Quality**: Reduce FPS to 24 (smaller file)
- **Low Quality**: Reduce FPS to 15 (smallest file)

## Troubleshooting

### Common Issues

#### 1. "Image not found" Error
- Check file path in script
- Ensure image is in `context/` folder
- Verify file extension is correct

#### 2. "Audio not found" Error  
- Check audio file path
- Ensure MP3 file is in `context/` folder
- Verify audio file is not corrupted

#### 3. Slow Processing
- Reduce image resolution
- Lower FPS setting
- Close other applications to free RAM

#### 4. Large File Size
- Reduce FPS from 30 to 24 or 15
- Compress input image
- Use shorter audio clips

### Performance Tips

1. **Optimize Images**: Use compressed JPG files
2. **Manage RAM**: Close other applications during processing
3. **Batch Processing**: Process multiple files overnight
4. **Quality vs Speed**: Balance FPS with file size needs

## Examples

### Basic Usage
```bash
# 1. Place files
cp my_image.jpg context/
cp my_song.mp3 context/

# 2. Edit lyrics in script
# 3. Run
python smooth_video_generator.py

# 4. Check output
ls context/*.mp4
```

### Custom Configuration
```python
# Modify script for custom settings
lyrics = ["Custom", "Lyrics", "Here"]
fps = 24  # Lower FPS for smaller files
zoom_variation = 0.2  # Gentler zoom
```

## Best Practices

1. **Test First**: Use short audio clips for testing
2. **Backup Files**: Keep original files safe
3. **Monitor Progress**: Watch console output for issues
4. **Quality Check**: Review output videos before batch processing
5. **File Management**: Organize input/output files properly
