import os
import time
from typing import Dict, List, Optional
from PIL import Image
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip
import logging

from .hardware_detector import HardwareDetector
from .model_manager import ModelManager
from .audio_processor import AudioProcessor

class VideoGenerator:
    """Main video generation class with dynamic hardware adaptation"""
    
    def __init__(self, hardware_tier: Optional[str] = None, 
                 model_preference: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize hardware detection
        self.hardware_detector = HardwareDetector()
        
        # Determine hardware tier
        self.hardware_tier = hardware_tier or self.hardware_detector.get_hardware_tier()
        
        # Initialize model manager
        self.model_manager = ModelManager(self.hardware_detector)
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor()
        
        # Load optimal model
        self.model_name = self.model_manager.select_optimal_model(
            self.hardware_tier, 
            [model_preference] if model_preference else None
        )
        
        self.model = None
        self.logger.info(f"Initialized with hardware tier: {self.hardware_tier}")
        self.logger.info(f"Selected model: {self.model_name}")
    
    def generate_video(self, image_path: str, audio_path: str, 
                     output_path: str, **kwargs) -> str:
        """Generate video from image and audio"""
        try:
            # Load and preprocess inputs
            image = self._load_image(image_path)
            audio_duration = self._get_audio_duration(audio_path)
            
            # Load model if not loaded
            if self.model is None:
                self.model = self.model_manager.load_model(self.model_name)
            
            # Generate video frames
            self.logger.info("Generating video frames...")
            start_time = time.time()
            
            frames = self.model.generate_video(
                image, 
                duration=audio_duration,
                **kwargs
            )
            
            generation_time = time.time() - start_time
            self.logger.info(f"Frame generation completed in {generation_time:.2f}s")
            
            # Create video from frames
            self.logger.info("Creating video file...")
            self._create_video_file(frames, audio_path, output_path)
            
            self.logger.info(f"Video generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Video generation failed: {e}")
            raise
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        audio = AudioFileClip(audio_path)
        duration = audio.duration
        audio.close()
        
        return duration
    
    def _create_video_file(self, frames: List[Image.Image], 
                          audio_path: str, output_path: str):
        """Create video file from frames and audio"""
        # Get optimal FPS
        fps = self.hardware_detector._get_max_fps(self.hardware_tier)
        
        # Create video clip from frames
        video_clip = ImageSequenceClip(frames, fps=fps)
        
        # Add audio
        audio_clip = AudioFileClip(audio_path)
        final_video = video_clip.set_audio(audio_clip)
        
        # Export video
        final_video.write_videofile(
            output_path,
            fps=fps,
            codec='libx264',
            audio_codec='aac'
        )
        
        # Cleanup
        video_clip.close()
        audio_clip.close()
        final_video.close()
    
    def get_estimated_time(self, duration: float) -> float:
        """Get estimated processing time"""
        time_per_second = self.hardware_detector._get_processing_time(self.hardware_tier)
        return duration * time_per_second
    
    def get_hardware_info(self) -> Dict:
        """Get hardware information"""
        return self.hardware_detector.get_hardware_summary()
    
    def get_capabilities(self) -> Dict:
        """Get current capabilities"""
        return self.hardware_detector.estimate_capabilities()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.model:
            self.model.cleanup()
        self.model_manager.cleanup_models()
