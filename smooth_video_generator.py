#!/usr/bin/env python3
"""
Simple MP3 + Image + Lyrics Video Generator
Gabungkan MP3 dengan gambar bergerak dan lyric yang sinkron
"""

import os
import sys
import logging
from PIL import Image, ImageDraw, ImageFont
import moviepy.editor as mp
from moviepy.editor import ImageSequenceClip, AudioFileClip
import librosa
import numpy as np
from typing import List, Tuple
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmoothVideoGenerator:
    """Smooth video generator dengan MP3 + gambar bergerak + lyric sinkron"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_smooth_animation(self, image_path: str, audio_path: str, lyrics: List[str], 
                               output_path: str = "output_video.mp4"):
        """Create smooth video dengan gambar bergerak dan lyric sinkron"""
        
        self.logger.info("Creating smooth video with MP3 + moving image + synchronized lyrics...")
        
        # Load image and audio
        image = Image.open(image_path)
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        
        self.logger.info(f"Audio duration: {duration:.2f}s")
        
        # Create smooth moving frames with synchronized lyrics
        frames = self._create_smooth_frames_with_lyrics(image, duration, lyrics, fps=30)
        
        # Convert PIL Images to numpy arrays
        numpy_frames = [np.array(frame) for frame in frames]
        
        # Create video clip
        video_clip = ImageSequenceClip(numpy_frames, fps=30)
        
        # Add audio
        final_video = video_clip.set_audio(audio_clip)
        
        # Export video
        final_video.write_videofile(
            output_path,
            fps=30,
            codec='libx264',
            audio_codec='aac'
        )
        
        self.logger.info(f"✅ Video created: {output_path}")
        return output_path
    
    def _create_smooth_frames_with_lyrics(self, image: Image.Image, duration: float, 
                                        lyrics: List[str], fps: int = 30) -> List:
        """Create smooth moving frames dengan efek smooth dan lyric sinkron"""
        
        frames = []
        total_frames = int(duration * fps)
        
        # Resize image untuk efek smooth
        width, height = image.size
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 60)
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 60)
            except:
                font = ImageFont.load_default()
        
        # Calculate lyric timing - lebih akurat
        if lyrics:
            # Buat timing yang lebih natural
            lyric_timings = []
            total_lyric_time = duration * 0.8  # 80% dari durasi untuk lyrics
            time_per_lyric = total_lyric_time / len(lyrics)
            
            for i in range(len(lyrics)):
                start_time = i * time_per_lyric
                end_time = (i + 1) * time_per_lyric
                lyric_timings.append((start_time, end_time))
        else:
            lyric_timings = []
        
        for i in range(total_frames):
            progress = i / total_frames
            time_in_video = i / fps
            
            # Smooth zoom effect dengan easing
            zoom_base = 1.0
            zoom_variation = 0.3
            zoom_factor = zoom_base + zoom_variation * math.sin(progress * math.pi * 3) * 0.5
            
            # Smooth pan effect dengan multiple frequencies
            pan_x = int(80 * math.sin(progress * math.pi * 2.5) + 40 * math.sin(progress * math.pi * 7))
            pan_y = int(60 * math.cos(progress * math.pi * 1.8) + 30 * math.sin(progress * math.pi * 5))
            
            # Smooth rotation effect (subtle)
            rotation_angle = 2 * math.sin(progress * math.pi * 1.5)  # Max 2 degrees
            
            # Apply transformations
            new_width = int(width * zoom_factor)
            new_height = int(height * zoom_factor)
            
            # Resize image dengan smooth interpolation
            zoomed_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Apply rotation
            if abs(rotation_angle) > 0.1:
                zoomed_image = zoomed_image.rotate(rotation_angle, expand=True, fillcolor=(0, 0, 0))
            
            # Create frame dengan smooth pan
            frame = Image.new("RGB", (width, height), (0, 0, 0))
            
            # Calculate paste position dengan smooth movement
            paste_x = (width - zoomed_image.width) // 2 + pan_x
            paste_y = (height - zoomed_image.height) // 2 + pan_y
            
            # Ensure paste position is within bounds
            paste_x = max(0, min(paste_x, width - zoomed_image.width))
            paste_y = max(0, min(paste_y, height - zoomed_image.height))
            
            frame.paste(zoomed_image, (paste_x, paste_y))
            
            # Add synchronized lyric text
            if lyrics and lyric_timings:
                current_lyric = None
                for j, (start_time, end_time) in enumerate(lyric_timings):
                    if start_time <= time_in_video <= end_time:
                        current_lyric = lyrics[j]
                        break
                
                if current_lyric:
                    # Draw text on frame dengan smooth appearance
                    draw = ImageDraw.Draw(frame)
                    
                    # Get text size
                    bbox = draw.textbbox((0, 0), current_lyric, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # Position text at bottom center dengan smooth positioning
                    text_x = (width - text_width) // 2
                    text_y = height - text_height - 80
                    
                    # Smooth text appearance effect
                    fade_progress = (time_in_video - start_time) / (end_time - start_time)
                    alpha = min(1.0, fade_progress * 3)  # Fade in quickly
                    
                    # Draw text dengan smooth outline dan glow effect
                    for offset in range(3, 0, -1):
                        alpha_offset = alpha * (1 - offset/3)
                        color_intensity = int(255 * alpha_offset)
                        
                        # Black outline
                        draw.text((text_x-offset, text_y-offset), current_lyric, font=font, fill=(0, 0, 0))
                        draw.text((text_x+offset, text_y-offset), current_lyric, font=font, fill=(0, 0, 0))
                        draw.text((text_x-offset, text_y+offset), current_lyric, font=font, fill=(0, 0, 0))
                        draw.text((text_x+offset, text_y+offset), current_lyric, font=font, fill=(0, 0, 0))
                    
                    # White text dengan smooth alpha
                    text_color = (int(255 * alpha), int(255 * alpha), int(255 * alpha))
                    draw.text((text_x, text_y), current_lyric, font=font, fill=text_color)
            
            frames.append(frame)
            
            if i % 150 == 0:
                self.logger.info(f"Creating smooth frames: {i/total_frames:.1%}")
        
        return frames

def main():
    """Main function"""
    
    logger.info("=== Smooth MP3 + Image + Lyrics Video Generator ===")
    
    # File paths
    image_path = "context/rapper.jpg"
    audio_path = "context/Title _ Judul_Black Chains _ Rantai Hi_cmp.mp3"
    output_path = "context/smooth_video_with_lyrics.mp4"
    
    # Check files
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return
    
    if not os.path.exists(audio_path):
        logger.error(f"Audio not found: {audio_path}")
        return
    
    # Sample lyrics dengan timing yang lebih natural
    lyrics = [
        "Black Chains",
        "Rantai Hitam",
        "Mengikat jiwa",
        "Dalam kegelapan",
        "Mencari cahaya",
        "Di tengah malam",
        "Black Chains",
        "Rantai Hitam",
        "Mengikat jiwa",
        "Dalam kegelapan",
        "Mencari cahaya",
        "Di tengah malam",
        "Black Chains",
        "Rantai Hitam",
        "Mengikat jiwa",
        "Dalam kegelapan"
    ]
    
    # Initialize generator
    generator = SmoothVideoGenerator()
    
    # Create video
    try:
        result = generator.create_smooth_animation(
            image_path=image_path,
            audio_path=audio_path,
            lyrics=lyrics,
            output_path=output_path
        )
        
        logger.info(f"✅ Video created successfully: {result}")
        
    except Exception as e:
        logger.error(f"Error creating video: {e}")

if __name__ == "__main__":
    main()

