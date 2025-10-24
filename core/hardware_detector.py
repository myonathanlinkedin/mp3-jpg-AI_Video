import torch
import psutil
import platform
import subprocess
from typing import Dict, List, Optional
import logging

class HardwareDetector:
    """Dynamic hardware detection and capability assessment"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gpu_info = self._detect_gpu()
        self.cpu_info = self._detect_cpu()
        self.ram_info = self._detect_ram()
        self.storage_info = self._detect_storage()
    
    def _detect_gpu(self) -> Dict:
        """Detect GPU capabilities"""
        gpu_info = {
            "available": False,
            "name": "Unknown",
            "vram_gb": 0,
            "compute_capability": "0.0",
            "cuda_version": None,
            "driver_version": None
        }
        
        try:
            if torch.cuda.is_available():
                gpu_info["available"] = True
                gpu_info["name"] = torch.cuda.get_device_name(0)
                gpu_info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
                gpu_info["compute_capability"] = torch.cuda.get_device_capability(0)
                gpu_info["cuda_version"] = torch.version.cuda
                
                # Get driver version
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                          capture_output=True, text=True)
                    gpu_info["driver_version"] = result.stdout.strip()
                except:
                    pass
                    
        except Exception as e:
            self.logger.warning(f"GPU detection failed: {e}")
            
        return gpu_info
    
    def _detect_cpu(self) -> Dict:
        """Detect CPU capabilities"""
        cpu_info = {
            "name": platform.processor(),
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            "architecture": platform.machine(),
            "usage_percent": psutil.cpu_percent(interval=1)
        }
        return cpu_info
    
    def _detect_ram(self) -> Dict:
        """Detect RAM capabilities"""
        ram_info = {
            "total_gb": psutil.virtual_memory().total / 1e9,
            "available_gb": psutil.virtual_memory().available / 1e9,
            "used_gb": psutil.virtual_memory().used / 1e9,
            "usage_percent": psutil.virtual_memory().percent
        }
        return ram_info
    
    def _detect_storage(self) -> Dict:
        """Detect storage capabilities"""
        storage_info = {
            "total_gb": 0,
            "free_gb": 0,
            "usage_percent": 0
        }
        
        try:
            disk_usage = psutil.disk_usage('/')
            storage_info["total_gb"] = disk_usage.total / 1e9
            storage_info["free_gb"] = disk_usage.free / 1e9
            storage_info["usage_percent"] = (disk_usage.used / disk_usage.total) * 100
        except:
            pass
            
        return storage_info
    
    def get_hardware_tier(self) -> str:
        """Determine hardware tier based on capabilities"""
        if not self.gpu_info["available"]:
            return "low"
        
        vram_gb = self.gpu_info["vram_gb"]
        ram_gb = self.ram_info["total_gb"]
        
        # Ultra tier: RTX 4090 Mobile (16GB VRAM)
        if vram_gb >= 16 and ram_gb >= 32:
            return "ultra"
        
        # High tier: RTX 4080/3080 (10-16GB VRAM)
        elif vram_gb >= 10 and ram_gb >= 16:
            return "high"
        
        # Medium tier: RTX 3060/4060 (8-12GB VRAM)
        elif vram_gb >= 8 and ram_gb >= 8:
            return "medium"
        
        # Low tier: Integrated graphics or CPU
        else:
            return "low"
    
    def estimate_capabilities(self) -> Dict:
        """Estimate processing capabilities"""
        tier = self.get_hardware_tier()
        
        capabilities = {
            "tier": tier,
            "max_resolution": self._get_max_resolution(tier),
            "max_fps": self._get_max_fps(tier),
            "max_duration": self._get_max_duration(tier),
            "processing_time_per_second": self._get_processing_time(tier),
            "supported_models": self._get_supported_models(tier)
        }
        
        return capabilities
    
    def _get_max_resolution(self, tier: str) -> str:
        """Get maximum supported resolution"""
        resolutions = {
            "ultra": "4K (3840x2160)",
            "high": "1080p (1920x1080)",
            "medium": "720p (1280x720)",
            "low": "480p (854x480)"
        }
        return resolutions.get(tier, "480p")
    
    def _get_max_fps(self, tier: str) -> int:
        """Get maximum supported FPS"""
        fps = {
            "ultra": 60,
            "high": 30,
            "medium": 24,
            "low": 15
        }
        return fps.get(tier, 15)
    
    def _get_max_duration(self, tier: str) -> int:
        """Get maximum recommended duration in seconds"""
        duration = {
            "ultra": 60,
            "high": 30,
            "medium": 15,
            "low": 10
        }
        return duration.get(tier, 10)
    
    def _get_processing_time(self, tier: str) -> float:
        """Get estimated processing time per second of video"""
        time_per_second = {
            "ultra": 0.3,  # 30 seconds for 10 seconds of video
            "high": 0.5,   # 50 seconds for 10 seconds of video
            "medium": 1.0, # 100 seconds for 10 seconds of video
            "low": 2.0     # 200 seconds for 10 seconds of video
        }
        return time_per_second.get(tier, 2.0)
    
    def _get_supported_models(self, tier: str) -> List[str]:
        """Get supported models for hardware tier"""
        models = {
            "ultra": ["stable_video_diffusion_xl", "runwayml_gen2", "animate_diff"],
            "high": ["stable_video_diffusion", "animate_diff", "ken_burns"],
            "medium": ["animate_diff", "ken_burns", "basic_animation"],
            "low": ["ken_burns", "basic_animation", "slideshow"]
        }
        return models.get(tier, ["slideshow"])
    
    def get_hardware_summary(self) -> Dict:
        """Get complete hardware summary"""
        return {
            "gpu": self.gpu_info,
            "cpu": self.cpu_info,
            "ram": self.ram_info,
            "storage": self.storage_info,
            "tier": self.get_hardware_tier(),
            "capabilities": self.estimate_capabilities()
        }
    
    def is_model_supported(self, model_name: str) -> bool:
        """Check if model is supported on current hardware"""
        tier = self.get_hardware_tier()
        supported_models = self._get_supported_models(tier)
        return model_name in supported_models
    
    def get_optimal_settings(self, model_name: str) -> Dict:
        """Get optimal settings for specific model"""
        tier = self.get_hardware_tier()
        
        settings = {
            "batch_size": self._get_batch_size(tier, model_name),
            "precision": self._get_precision(tier),
            "chunk_size": self._get_chunk_size(tier),
            "max_resolution": self._get_max_resolution(tier),
            "max_fps": self._get_max_fps(tier)
        }
        
        return settings
    
    def _get_batch_size(self, tier: str, model_name: str) -> int:
        """Get optimal batch size"""
        if tier == "ultra":
            return 4
        elif tier == "high":
            return 2
        elif tier == "medium":
            return 1
        else:
            return 1
    
    def _get_precision(self, tier: str) -> str:
        """Get optimal precision"""
        if tier in ["ultra", "high"]:
            return "fp16"
        else:
            return "fp32"
    
    def _get_chunk_size(self, tier: str) -> int:
        """Get optimal chunk size for processing"""
        if tier == "ultra":
            return 100
        elif tier == "high":
            return 50
        elif tier == "medium":
            return 25
        else:
            return 10
