from typing import Dict, List, Optional
import torch
from .hardware_detector import HardwareDetector
from .base_model import BaseModel
import logging

class ModelManager:
    """Dynamic model management based on hardware capabilities"""
    
    def __init__(self, hardware_detector: HardwareDetector):
        self.hardware_detector = hardware_detector
        self.logger = logging.getLogger(__name__)
        self.loaded_models = {}
        
    def select_optimal_model(self, hardware_tier: str, preferences: List[str] = None) -> str:
        """Select optimal model based on hardware tier and preferences"""
        supported_models = self.hardware_detector._get_supported_models(hardware_tier)
        
        if preferences:
            # Try to match preferences
            for pref in preferences:
                if pref in supported_models:
                    return pref
        
        # Default selection based on tier
        if hardware_tier == "ultra":
            return "stable_video_diffusion_xl"
        elif hardware_tier == "high":
            return "stable_video_diffusion"
        elif hardware_tier == "medium":
            return "animate_diff"
        else:
            return "ken_burns"
    
    def load_model(self, model_name: str) -> BaseModel:
        """Load model with hardware optimization"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # Get optimal settings
        settings = self.hardware_detector.get_optimal_settings(model_name)
        
        # Force Ken Burns for now (no AI model dependency)
        if model_name in ["stable_video_diffusion_xl", "stable_video_diffusion", "animate_diff"]:
            self.logger.warning(f"AI model {model_name} requires authentication, falling back to Ken Burns")
            model_name = "ken_burns"
        
        # Load model based on name
        if model_name == "ken_burns":
            model = self._load_ken_burns(settings)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.loaded_models[model_name] = model
        return model
    
    def _load_stable_video_diffusion_xl(self, settings: Dict) -> BaseModel:
        """Load Stable Video Diffusion XL with optimization"""
        try:
            from diffusers import StableVideoDiffusionPipeline
            
            # Load with optimal settings
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion",
                torch_dtype=torch.float16 if settings["precision"] == "fp16" else torch.float32,
                variant="fp16" if settings["precision"] == "fp16" else None
            )
            
            # Move to GPU if available
            if self.hardware_detector.gpu_info["available"]:
                pipe = pipe.to("cuda")
            
            return pipe
            
        except Exception as e:
            self.logger.error(f"Failed to load SVD XL: {e}")
            raise
    
    def _load_stable_video_diffusion(self, settings: Dict) -> BaseModel:
        """Load Stable Video Diffusion with optimization"""
        try:
            from diffusers import StableVideoDiffusionPipeline
            
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion",
                torch_dtype=torch.float16 if settings["precision"] == "fp16" else torch.float32
            )
            
            if self.hardware_detector.gpu_info["available"]:
                pipe = pipe.to("cuda")
            
            return pipe
            
        except Exception as e:
            self.logger.error(f"Failed to load SVD: {e}")
            raise
    
    def _load_animate_diff(self, settings: Dict) -> BaseModel:
        """Load AnimateDiff with optimization"""
        try:
            from diffusers import AnimateDiffPipeline, MotionAdapter
            
            adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
            pipe = AnimateDiffPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                motion_adapter=adapter,
                torch_dtype=torch.float16 if settings["precision"] == "fp16" else torch.float32
            )
            
            if self.hardware_detector.gpu_info["available"]:
                pipe = pipe.to("cuda")
            
            return pipe
            
        except Exception as e:
            self.logger.error(f"Failed to load AnimateDiff: {e}")
            raise
    
    def _load_ken_burns(self, settings: Dict) -> BaseModel:
        """Load Ken Burns effect (CPU-based)"""
        try:
            from .ken_burns import KenBurnsModel
            return KenBurnsModel(settings)
            
        except Exception as e:
            self.logger.error(f"Failed to load Ken Burns: {e}")
            raise
    
    def optimize_model_for_hardware(self, model: BaseModel, hardware_info: Dict) -> BaseModel:
        """Optimize model for specific hardware"""
        # Enable optimizations based on hardware
        if hardware_info["gpu"]["available"]:
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
        
        # Set memory management
        if hasattr(model, 'enable_memory_efficient_attention'):
            model.enable_memory_efficient_attention()
        
        return model
    
    def get_model_requirements(self, model_name: str) -> Dict:
        """Get model requirements"""
        requirements = {
            "stable_video_diffusion_xl": {
                "min_vram_gb": 16,
                "min_ram_gb": 32,
                "min_storage_gb": 20,
                "cuda_required": True
            },
            "stable_video_diffusion": {
                "min_vram_gb": 8,
                "min_ram_gb": 16,
                "min_storage_gb": 15,
                "cuda_required": True
            },
            "animate_diff": {
                "min_vram_gb": 6,
                "min_ram_gb": 8,
                "min_storage_gb": 10,
                "cuda_required": False
            },
            "ken_burns": {
                "min_vram_gb": 0,
                "min_ram_gb": 4,
                "min_storage_gb": 1,
                "cuda_required": False
            }
        }
        
        return requirements.get(model_name, {})
    
    def check_model_compatibility(self, model_name: str) -> bool:
        """Check if model is compatible with current hardware"""
        requirements = self.get_model_requirements(model_name)
        hardware_info = self.hardware_detector.get_hardware_summary()
        
        # Check VRAM
        if requirements.get("min_vram_gb", 0) > hardware_info["gpu"]["vram_gb"]:
            return False
        
        # Check RAM
        if requirements.get("min_ram_gb", 0) > hardware_info["ram"]["total_gb"]:
            return False
        
        # Check CUDA requirement
        if requirements.get("cuda_required", False) and not hardware_info["gpu"]["available"]:
            return False
        
        return True
    
    def cleanup_models(self):
        """Cleanup loaded models to free memory"""
        for model_name, model in self.loaded_models.items():
            if hasattr(model, 'cpu'):
                model.cpu()
            del model
        
        self.loaded_models.clear()
        torch.cuda.empty_cache()
