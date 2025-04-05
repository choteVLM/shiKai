#!/usr/bin/env python3
"""
Standalone script to process video frame descriptions into world state history.
This script has no dependencies on PyTorch or CUDA and is just for testing the form_world_state_history function.
"""

import re
import os
import json
import logging
import traceback
import numpy as np
from typing import List, Dict, Tuple, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("world_state_processor")

def convert_to_free_form_text(world_state_history: List[Dict[str, Any]]) -> str:
    """
    Converts the world state history to a free-form text representation.
    Matches the exact format in form_world_state_history.py
    
    Args:
        world_state_history: List of dictionaries containing world state for each interval
        
    Returns:
        String containing a free-form text representation of the world state history
    """
    free_form_text = ""
    
    for interval in world_state_history:
        free_form_text += f"**Timestamp**: {interval['time stamp']}\n"
        
        for key, value in interval.items():
            if key != 'time stamp':
                free_form_text += f"**{key}**: {value}\n"
        
        free_form_text += "\n"
    
    return free_form_text

def convert_seconds_to_hms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def parse_frame_descriptions(file_path: str) -> Tuple[List[Tuple[float, float]], List[str]]:
    """
    Parse frame descriptions from a text file
    
    Args:
        file_path: Path to the text file containing frame descriptions
        
    Returns:
        Tuple of (timestamps, descriptions)
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract frame descriptions using regex pattern matching
    pattern = r"Time: (\d+\.\d+)s - (\d+\.\d+)s\nDescription: (.*?)(?=\n--+\n|$)"
    matches = re.findall(pattern, content, re.DOTALL)
    
    if not matches:
        raise ValueError(f"Failed to parse frame descriptions from {file_path}")
    
    # Extract descriptions and timestamps
    frame_timestamps = []
    frame_descriptions = []
    
    for start_time, end_time, description in matches:
        frame_timestamps.append((float(start_time), float(end_time)))
        frame_descriptions.append(description.strip())
    
    return frame_timestamps, frame_descriptions

def create_world_state_history(
    frame_timestamps: List[Tuple[float, float]], 
    frame_descriptions: List[str], 
    frame_interval_seconds: int = 10
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Creates a world state history from frame descriptions and timestamps
    
    Args:
        frame_timestamps: List of (start_time, end_time) tuples in seconds
        frame_descriptions: List of frame descriptions
        frame_interval_seconds: Time interval in seconds for grouping frames
        
    Returns:
        Tuple of (world_state_history, free_form_text)
    """
    # Calculate video duration from last timestamp
    duration_seconds = frame_timestamps[-1][1] if frame_timestamps else 0
    
    # Calculate number of intervals
    num_intervals = int(np.ceil(duration_seconds / frame_interval_seconds))
    
    # Create world state history
    world_state_history = []
    
    for interval_idx in range(num_intervals):
        start_second = interval_idx * frame_interval_seconds
        end_second = min((interval_idx + 1) * frame_interval_seconds, duration_seconds)
        
        start_time = convert_seconds_to_hms(start_second)
        end_time = convert_seconds_to_hms(end_second)
        
        # Find frames that belong to this interval
        interval_frames = []
        for i, (frame_start, frame_end) in enumerate(frame_timestamps):
            if frame_start >= start_second and frame_start < end_second:
                interval_frames.append(frame_descriptions[i])
        
        # Create summary for this interval
        if interval_frames:
            # Combine descriptions into a comprehensive scene description
            combined_description = " ".join(interval_frames)
            
            # Create dictionary for this interval - simplified to match form_world_state_history.py
            interval_dict = {
                "time stamp": f"{start_time} - {end_time}",
                "scene description": combined_description
            }
            
            world_state_history.append(interval_dict)
    
    # Create free-form text representation
    free_form_text = convert_to_free_form_text(world_state_history)
    
    return world_state_history, free_form_text

def main():
    """Main function to process video frame descriptions"""
    try:
        # Parse the pre-generated frame descriptions file
        frame_descriptions_file = "video_frames_20250328_210734.txt"
        logger.info(f"Parsing frame descriptions from {frame_descriptions_file}")
        
        # Parse frame descriptions and timestamps
        frame_timestamps, frame_descriptions = parse_frame_descriptions(frame_descriptions_file)
        logger.info(f"Successfully parsed {len(frame_descriptions)} frame descriptions")
        
        # Calculate video duration
        duration_seconds = frame_timestamps[-1][1] if frame_timestamps else 0
        logger.info(f"Video duration: {convert_seconds_to_hms(duration_seconds)} ({duration_seconds:.2f}s)")
        
        # Create world state history with 10-second intervals
        logger.info("Creating world state history...")
        world_state_history, free_form_text = create_world_state_history(
            frame_timestamps, frame_descriptions, frame_interval_seconds=10
        )
        logger.info(f"Created {len(world_state_history)} world state intervals")
        
        # Save the results
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        json_path = os.path.join(output_dir, "world_state_history.json")
        with open(json_path, "w") as f:
            json.dump(world_state_history, f, indent=2)
        
        # Save free-form text representation
        text_path = os.path.join(output_dir, "world_state_text.txt")
        with open(text_path, "w") as f:
            f.write(free_form_text)
        
        logger.info(f"World state history saved to {json_path}")
        logger.info(f"Free-form text representation saved to {text_path}")
        
        # Print a sample of the world state history
        num_samples = min(3, len(world_state_history))
        logger.info(f"Sample of {num_samples} world state entries:")
        for i in range(num_samples):
            print(f"\nInterval {i+1}:")
            print(f"Time stamp: {world_state_history[i]['time stamp']}")
            print(f"Scene description: {world_state_history[i]['scene description'][:150]}...")
        
    except Exception as e:
        logger.error(f"Error processing world state history: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 