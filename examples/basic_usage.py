#!/usr/bin/env python3
"""
Basic usage example for AI Video Generator
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.video_generator import VideoGenerator
from core.hardware_detector import HardwareDetector

def extract_audio_from_video(video_path: str, output_audio_path: str) -> str:
    """Extract audio from MP4 video file"""
    try:
        from moviepy.editor import VideoFileClip
        
        # Load video
        video = VideoFileClip(video_path)
        
        # Extract audio
        audio = video.audio
        
        # Write audio file
        audio.write_audiofile(output_audio_path, verbose=False, logger=None)
        
        # Cleanup
        audio.close()
        video.close()
        
        return output_audio_path
        
    except Exception as e:
        logging.error(f"Failed to extract audio: {e}")
        raise

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize hardware detector
        logger.info("Initializing hardware detector...")
        detector = HardwareDetector()
        
        # Get hardware information
        hardware_info = detector.get_hardware_summary()
        logger.info(f"Hardware tier: {hardware_info['tier']}")
        logger.info(f"GPU: {hardware_info['gpu']['name']}")
        logger.info(f"VRAM: {hardware_info['gpu']['vram_gb']:.1f}GB")
        logger.info(f"RAM: {hardware_info['ram']['total_gb']:.1f}GB")
        
        # Initialize video generator with Ken Burns fallback
        logger.info("Initializing video generator...")
        generator = VideoGenerator(model_preference="ken_burns")
        
        # Get capabilities
        capabilities = generator.get_capabilities()
        logger.info(f"Max resolution: {capabilities['max_resolution']}")
        logger.info(f"Max FPS: {capabilities['max_fps']}")
        logger.info(f"Supported models: {capabilities['supported_models']}")
        
        # Use context files directly
        image_path = "context/rapper.jpg"
        audio_path = "context/Title _ Judul_Black Chains _ Rantai Hi_cmp.mp3"
        output_path = "context/generated_rapper_video.mp4"
        
        # Check if input files exist
        if not os.path.exists(image_path):
            logger.warning(f"Image file not found: {image_path}")
            logger.info("Please provide a valid image file")
            return
        
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            logger.info("Please provide a valid audio file")
            return
        
        # Generate video
        logger.info("Starting video generation...")
        logger.info(f"Input image: {image_path}")
        logger.info(f"Input audio: {audio_path}")
        logger.info(f"Output video: {output_path}")
        
        # Estimate processing time
        audio_duration = generator.audio_processor.get_audio_duration(audio_path)
        estimated_time = generator.get_estimated_time(audio_duration)
        logger.info(f"Audio duration: {audio_duration:.2f}s")
        logger.info(f"Estimated processing time: {estimated_time:.2f}s")
        
        # Generate video
        result_path = generator.generate_video(
            image_path=image_path,
            audio_path=audio_path,
            output_path=output_path
        )
        
        logger.info(f"Video generated successfully: {result_path}")
        
        # Cleanup
        generator.cleanup()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
