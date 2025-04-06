"""
Video processing utilities for SmolVLM.

This module contains utilities for extracting and processing video frames.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List
import logging

logger = logging.getLogger(__name__)

class VideoFrameExtractor:
    """
    A utility class for extracting frames from videos.
    
    This class handles frame extraction, resizing, and center cropping
    to prepare frames for processing by vision-language models.
    """
    
    def __init__(self, max_frames: int = 50):
        """
        Initialize the VideoFrameExtractor.
        
        Args:
            max_frames: Maximum number of frames to extract from a video
        """
        self.max_frames = max_frames
        
    def resize_and_center_crop(self, image: Image.Image, target_size: int) -> Image.Image:
        """
        Resize and center crop an image to the target size.
        
        Args:
            image: PIL Image to process
            target_size: Target size for the image (square)
            
        Returns:
            Processed PIL Image
        """
        # Get current dimensions
        width, height = image.size
        
        # Calculate new dimensions keeping aspect ratio
        if width < height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))
            
        # Resize
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Center crop
        left = (new_width - target_size) // 2
        top = (new_height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        
        return image.crop((left, top, right, bottom))
    
    def extract_frames_at_fps(self, video_path: str, sample_fps: int, crop_frames: bool = True, target_size: int = 384) -> List[Image.Image]:
        """
        Extract frames from a video file at a specified sampling rate with optional cropping.
        
        Args:
            video_path: Path to the video file
            sample_fps: Frames per second to sample (e.g., 1 for one frame per second)
            crop_frames: Whether to resize and center crop frames (default: True)
            target_size: Target size for cropping if crop_frames is True (default: 384)
            
        Returns:
            List of PIL Images representing the extracted frames
            
        Raises:
            ValueError: If the video cannot be opened
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Video properties: {total_frames} frames, {original_fps} FPS")
        
        # Calculate sampling interval based on requested fps
        if sample_fps <= 0:
            sample_fps = 1  # Default to 1 fps if invalid value provided
            
        frame_interval = int(original_fps / sample_fps)
        frame_indices = list(range(0, total_frames, frame_interval))
        
        # If we have more frames than max_frames, sample evenly
        if len(frame_indices) > self.max_frames:
            indices = np.linspace(0, len(frame_indices) - 1, self.max_frames, dtype=int)
            frame_indices = [frame_indices[i] for i in indices]
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame)
                
                # Apply center crop if requested
                if crop_frames:
                    pil_image = self.resize_and_center_crop(pil_image, target_size)
                
                frames.append(pil_image)
            else:
                logger.warning(f"Failed to read frame at index {frame_idx}")
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from video at {sample_fps} FPS")
        return frames

def get_video_metadata(video_path: str) -> dict:
    """
    Get metadata about a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary containing metadata (fps, frame_count, duration)
        
    Raises:
        ValueError: If the video cannot be opened
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps
    
    cap.release()
    
    return {
        "fps": fps,
        "frame_count": total_frames,
        "duration_seconds": duration_seconds
    } 