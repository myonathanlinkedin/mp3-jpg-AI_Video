from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import torch
from PIL import Image

class BaseModel(ABC):
    """Base interface for all video generation models"""
    
    def __init__(self, settings: Dict):
        self.settings = settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @abstractmethod
    def generate_video(self, image: Image.Image, **kwargs) -> List[Image.Image]:
        """Generate video frames from image"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict:
        """Get model information"""
        pass
    
    @abstractmethod
    def estimate_processing_time(self, duration: float) -> float:
        """Estimate processing time for given duration"""
        pass
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for model"""
        return image
    
    def postprocess_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Postprocess generated frames"""
        return frames
    
    def cleanup(self):
        """Cleanup model resources"""
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()
