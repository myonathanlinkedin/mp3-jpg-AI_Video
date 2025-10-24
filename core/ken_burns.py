import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, List
import logging

class KenBurnsModel:
    """Ken Burns effect implementation for low-end hardware"""
    
    def __init__(self, settings: Dict):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
    
    def generate_video(self, image: Image.Image, duration: float = 10.0, 
                     fps: int = 24, **kwargs) -> List[Image.Image]:
        """Generate video frames using Ken Burns effect"""
        try:
            total_frames = int(duration * fps)
            frames = []
            
            # Get effect parameters
            effect_type = kwargs.get('effect_type', 'zoom_pan')
            zoom_factor = kwargs.get('zoom_factor', 1.2)
            pan_distance = kwargs.get('pan_distance', 0.3)
            
            for frame_num in range(total_frames):
                progress = frame_num / total_frames
                
                if effect_type == 'zoom_pan':
                    frame = self._create_zoom_pan_frame(image, progress, zoom_factor, pan_distance)
                elif effect_type == 'zoom_in':
                    frame = self._create_zoom_in_frame(image, progress, zoom_factor)
                elif effect_type == 'zoom_out':
                    frame = self._create_zoom_out_frame(image, progress, zoom_factor)
                elif effect_type == 'pan_left':
                    frame = self._create_pan_frame(image, progress, 'left', pan_distance)
                elif effect_type == 'pan_right':
                    frame = self._create_pan_frame(image, progress, 'right', pan_distance)
                else:
                    frame = self._create_zoom_pan_frame(image, progress, zoom_factor, pan_distance)
                
                frames.append(frame)
            
            return frames
            
        except Exception as e:
            self.logger.error(f"Ken Burns generation failed: {e}")
            raise
    
    def _create_zoom_pan_frame(self, image: Image.Image, progress: float, 
                              zoom_factor: float, pan_distance: float) -> Image.Image:
        """Create frame with zoom and pan effect"""
        # Calculate zoom
        zoom = 1.0 + (progress * (zoom_factor - 1.0))
        
        # Calculate pan
        pan_x = progress * pan_distance
        pan_y = progress * pan_distance * 0.5
        
        return self._apply_transform(image, zoom, pan_x, pan_y)
    
    def _create_zoom_in_frame(self, image: Image.Image, progress: float, 
                             zoom_factor: float) -> Image.Image:
        """Create frame with zoom in effect"""
        zoom = 1.0 + (progress * (zoom_factor - 1.0))
        return self._apply_transform(image, zoom, 0, 0)
    
    def _create_zoom_out_frame(self, image: Image.Image, progress: float, 
                              zoom_factor: float) -> Image.Image:
        """Create frame with zoom out effect"""
        zoom = zoom_factor - (progress * (zoom_factor - 1.0))
        return self._apply_transform(image, zoom, 0, 0)
    
    def _create_pan_frame(self, image: Image.Image, progress: float, 
                         direction: str, pan_distance: float) -> Image.Image:
        """Create frame with pan effect"""
        if direction == 'left':
            pan_x = -progress * pan_distance
            pan_y = 0
        elif direction == 'right':
            pan_x = progress * pan_distance
            pan_y = 0
        elif direction == 'up':
            pan_x = 0
            pan_y = -progress * pan_distance
        elif direction == 'down':
            pan_x = 0
            pan_y = progress * pan_distance
        else:
            pan_x = 0
            pan_y = 0
        
        return self._apply_transform(image, 1.0, pan_x, pan_y)
    
    def _apply_transform(self, image: Image.Image, zoom: float, 
                        pan_x: float, pan_y: float) -> Image.Image:
        """Apply zoom and pan transform to image"""
        # Get image dimensions
        width, height = image.size
        
        # Calculate new dimensions
        new_width = int(width * zoom)
        new_height = int(height * zoom)
        
        # Resize image
        resized = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Calculate crop area
        crop_left = int((new_width - width) / 2 + pan_x * width)
        crop_top = int((new_height - height) / 2 + pan_y * height)
        crop_right = crop_left + width
        crop_bottom = crop_top + height
        
        # Ensure crop area is within bounds
        crop_left = max(0, min(crop_left, new_width - width))
        crop_top = max(0, min(crop_top, new_height - height))
        crop_right = crop_left + width
        crop_bottom = crop_top + height
        
        # Crop image
        cropped = resized.crop((crop_left, crop_top, crop_right, crop_bottom))
        
        return cropped
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "name": "Ken Burns Effect",
            "type": "traditional_animation",
            "description": "Ken Burns effect with zoom and pan",
            "hardware_requirements": {
                "min_vram_gb": 0,
                "min_ram_gb": 2,
                "cuda_required": False
            },
            "supported_effects": [
                "zoom_pan", "zoom_in", "zoom_out", 
                "pan_left", "pan_right", "pan_up", "pan_down"
            ]
        }
    
    def estimate_processing_time(self, duration: float) -> float:
        """Estimate processing time for given duration"""
        # Ken Burns is very fast
        return duration * 0.1  # 0.1 seconds per second of video
    
    def cleanup(self):
        """Cleanup model resources"""
        pass
