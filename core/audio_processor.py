import os
import librosa
import soundfile as sf
from typing import Dict, List, Optional
import logging

class AudioProcessor:
    """Audio processing utilities for video generation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            duration = librosa.get_duration(filename=audio_path)
            return duration
        except Exception as e:
            self.logger.error(f"Failed to get audio duration: {e}")
            raise
    
    def get_audio_info(self, audio_path: str) -> Dict:
        """Get audio file information"""
        try:
            info = sf.info(audio_path)
            return {
                "duration": info.duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "format": info.format,
                "subtype": info.subtype
            }
        except Exception as e:
            self.logger.error(f"Failed to get audio info: {e}")
            raise
    
    def convert_audio_format(self, input_path: str, output_path: str, 
                           target_format: str = "mp3") -> str:
        """Convert audio to target format"""
        try:
            # Load audio
            audio, sr = librosa.load(input_path)
            
            # Save in target format
            sf.write(output_path, audio, sr, format=target_format)
            
            return output_path
        except Exception as e:
            self.logger.error(f"Failed to convert audio: {e}")
            raise
    
    def normalize_audio(self, audio_path: str, output_path: str) -> str:
        """Normalize audio volume"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path)
            
            # Normalize
            audio_normalized = librosa.util.normalize(audio)
            
            # Save
            sf.write(output_path, audio_normalized, sr)
            
            return output_path
        except Exception as e:
            self.logger.error(f"Failed to normalize audio: {e}")
            raise
    
    def trim_audio(self, audio_path: str, output_path: str, 
                   start_time: float, end_time: float) -> str:
        """Trim audio to specified duration"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path)
            
            # Calculate samples
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Trim
            audio_trimmed = audio[start_sample:end_sample]
            
            # Save
            sf.write(output_path, audio_trimmed, sr)
            
            return output_path
        except Exception as e:
            self.logger.error(f"Failed to trim audio: {e}")
            raise
    
    def validate_audio_file(self, audio_path: str) -> bool:
        """Validate audio file"""
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                return False
            
            # Try to load audio
            librosa.load(audio_path)
            return True
            
        except Exception as e:
            self.logger.warning(f"Audio validation failed: {e}")
            return False
