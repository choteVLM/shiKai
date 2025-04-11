"""
Text formatting utilities.

This module contains utilities for formatting text, including prompt loading,
time formatting, and world state text creation.
"""

import yaml
import json
import logging
import cv2
import numpy as np
from typing import Dict, Optional, Union, List, Tuple, Any
from functools import partial

logger = logging.getLogger(__name__)

#
# Prompt formatting functions
#

def format_time_for_prompt(seconds: float) -> Dict[str, str]:
    """
    Format time values for better readability in prompts.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Dictionary with formatted time values
    """
    # Round to 2 decimal places
    seconds = round(seconds, 2)
    
    # For total duration, convert to minutes if over 60 seconds
    formatted_duration = ""
    if seconds >= 60:
        minutes = int(seconds // 60)
        remainder_seconds = seconds % 60
        if minutes == 1:
            if remainder_seconds == 0:
                formatted_duration = "1 minute"
            else:
                formatted_duration = f"1 minute and {int(remainder_seconds)} seconds"
        else:
            if remainder_seconds == 0:
                formatted_duration = f"{minutes} minutes"
            else:
                formatted_duration = f"{minutes} minutes and {int(remainder_seconds)} seconds"
    else:
        formatted_duration = f"{seconds} seconds"
    
    return {
        "seconds": str(seconds),
        "formatted": formatted_duration
    }

def load_question_prompt(
    prompt: str,
    frames_count: int = 30, 
    time_interval: float = 2.0, 
    total_duration: float = 60.0,
    frames_per_context: int = 1,
    extra_replacements: Optional[Dict[str, str]] = None
) -> str:
    """
    Loads a prompt from a YAML file and dynamically replaces placeholders with actual values.
    
    Args:
        yaml_file_path: Path to the YAML file containing the prompt
        frames_count: Number of frames extracted from the video
        time_interval: Time interval between frames in seconds
        total_duration: Total duration of the video in seconds
        frames_per_context: Number of frames in each context (for multi-frame mode)
        extra_replacements: Additional placeholders to replace
        
    Returns:
        The prompt string
        
    """

    
    # Format time values
    interval_time = format_time_for_prompt(time_interval)
    duration_time = format_time_for_prompt(total_duration)
    
    # For multi-frame contexts, the context interval is different than the individual frame interval
    context_interval = time_interval * frames_per_context
    context_interval_time = format_time_for_prompt(context_interval)
    
    # Replace placeholders with actual values
    prompt = partial(prompt.format , FRAMES_COUNT = frames_count,
                                   TIME_INTERVAL = interval_time["seconds"],
                                   TIME_INTERVAL_FORMATTED = interval_time["formatted"],
                                   TOTAL_DURATION = duration_time["seconds"],
                                   TOTAL_DURATION_FORMATTED = duration_time["formatted"],
                                   FRAMES_PER_CONTEXT = frames_per_context,
                                   CONTEXT_INTERVAL = context_interval_time["seconds"],
                                   CONTEXT_INTERVAL_FORMATTED = context_interval_time["formatted"]
                                   )
    # Handle any additional replacements
    if extra_replacements:
        for key, value in extra_replacements.items():
            prompt = prompt(**{key:value})
    else:
        prompt = prompt()
    
    return prompt

#
# World state text formatting functions
#

def convert_seconds_to_hms(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to extract JSON from text.
    
    Args:
        text: Text potentially containing JSON
        
    Returns:
        Parsed JSON dict or None if extraction fails
    """
    try:
        json_start = text.find('{')
        json_end = text.rfind('}')
        
        if json_start >= 0 and json_end > json_start:
            json_str = text[json_start:json_end+1]
            return json.loads(json_str)
        return None
    except json.JSONDecodeError:
        return None

def clean_model_response(response: str) -> str:
    """
    Cleans the model response by removing user instructions and prompt text.
    
    Args:
        response: Raw model response text
        
    Returns:
        Cleaned response text with only the model's answer
    """
    # Check for the Assistant: tag (with or without newline)
    assistant_tag = "Assistant:"
    assistant_index = response.find(assistant_tag)
    
    if assistant_index >= 0:
        # Return everything after the tag
        return response[assistant_index + len(assistant_tag):].strip()
    
    # If no pattern found, return original text
    return response

def extract_json_response(response: str) -> str:
    """
    Extracts only the JSON part from a response.
    
    Args:
        response: Model response text
        
    Returns:
        Extracted JSON content or original text if no JSON found
    """
    # Look for JSON pattern with triple backticks
    json_start = response.find("```json")
    if json_start >= 0:
        # Find the end of the JSON block
        json_end = response.find("```", json_start + 6)
        if json_end >= 0:
            # Extract the JSON content without the backticks
            json_content = response[json_start + 7:json_end].strip()
            return json_content
    
    # If response itself looks like a JSON object, return as is
    if response.strip().startswith('{') and response.strip().endswith('}'):
        return response.strip()
    
    return response

def create_world_state_history(
    video_path: str, 
    frame_descriptions: List[str], 
    frame_interval_seconds: int = 60,
    preserve_json_structure: bool = False
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Creates a world state history from frame descriptions, organizing them into time-stamped intervals.
    
    Args:
        video_path: Path to the video file
        frame_descriptions: List of descriptions for each frame
        frame_interval_seconds: Time interval in seconds for grouping frames (default: 60 seconds)
        preserve_json_structure: If True, attempt to parse JSON structure from frame descriptions
        
    Returns:
        Tuple containing:
        - List of dictionaries containing world state for each interval
        - Free-form text representation of the world state history
    """
    # Get video duration
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps
    cap.release()
    
    logger.info(f"Video metadata: {total_frames} frames, {fps} FPS, {duration_seconds:.2f}s duration")
    
    # Calculate number of intervals
    num_intervals = int(np.ceil(duration_seconds / frame_interval_seconds))
    logger.info(f"Creating {num_intervals} intervals of {frame_interval_seconds} seconds each")
    
    # Create world state history
    world_state_history = []
    
    # Clean up response text
    
    for interval_idx in range(num_intervals):
        start_second = interval_idx * frame_interval_seconds
        end_second = min((interval_idx + 1) * frame_interval_seconds, duration_seconds)  # -2 to avoid overlap
        
        start_time = convert_seconds_to_hms(start_second)
        end_time = convert_seconds_to_hms(end_second)
        
        # Find frames that belong to this interval
        interval_frames = []
        for i, desc in enumerate(frame_descriptions):
            # Calculate approximate timestamp for this frame
            frame_second = i * (duration_seconds / len(frame_descriptions))
            if start_second <= frame_second < end_second:
                interval_frames.append(desc)
        
        # Create summary for this interval
        if interval_frames:
            interval_dict = {
                "time_stamp": f"{start_time} - {end_time}",
            }
            
            if preserve_json_structure:
                # Try to parse JSON structure from descriptions
                try:
                    parsed_frames = []
                    for desc in interval_frames:
                        # Try to extract JSON from text
                        json_desc = extract_json_response(desc)
                        parsed_json = extract_json_from_text(json_desc)
                        if parsed_json:
                            parsed_frames.append(parsed_json)
                        else:
                            # If no JSON found, use the plain text
                            parsed_frames.append({"description": desc})
                    
                    # Merge the parsed frames
                    merged_dict = {}
                    for key in ["Scene Context", "Motion Description", "Spatial Relationship Analysis", 
                                "Detailed Object Analysis", "Temporal Relationship Context", 
                                "Additional Details", "Summary",
                                "Sequence Timeframe", "Motion Analysis", "Key Changes", 
                                "Stable Elements", "Micro-Narrative"]:
                        # Get all non-empty values for this key
                        values = [frame.get(key, "") for frame in parsed_frames if key in frame and frame[key]]
                        if values:
                            merged_dict[key] = " ".join(values)
                    
                    # If structured data was found, use it
                    if merged_dict:
                        interval_dict.update(merged_dict)
                    else:
                        # Otherwise fall back to plain text
                        interval_dict["scene_description"] = " ".join(interval_frames)
                except Exception as e:
                    logger.warning(f"Failed to parse JSON structure: {e}")
                    interval_dict["scene_description"] = " ".join(interval_frames)
            else:
                # Simple concatenation
                interval_dict["scene_description"] = " ".join(interval_frames)
            
            world_state_history.append(interval_dict)
    
    # Create free-form text representation
    free_form_text = convert_to_free_form_text(world_state_history)
    
    return world_state_history, free_form_text

def convert_to_free_form_text(world_state_history: List[Dict[str, Any]]) -> str:
    """
    Converts the world state history to a free-form text representation.
    
    Args:
        world_state_history: List of dictionaries containing world state for each interval
        
    Returns:
        String containing a free-form text representation of the world state history
    """
    free_form_text = ""
    
    for interval in world_state_history:
        free_form_text += f"**Timestamp**: {interval.get('time_stamp', '')}\n"
        
        for key, value in interval.items():
            if key != 'time_stamp':
                # Format key for display
                display_key = key.replace('_', ' ').title()
                free_form_text += f"**{display_key}**: {value}\n"
        
        free_form_text += "\n"
    
    return free_form_text

def merge_world_states(world_states: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple world states into a single consolidated world state.
    
    Args:
        world_states: List of world state dictionaries to merge
        
    Returns:
        Merged world state dictionary
    """
    if not world_states:
        return {}
    
    merged_state = {}
    
    # Determine timestamp range
    start_stamps = []
    end_stamps = []
    
    for state in world_states:
        if 'time_stamp' in state:
            try:
                start, end = state['time_stamp'].split(' - ')
                start_stamps.append(start)
                end_stamps.append(end)
            except:
                pass
    
    # Set the new timestamp range
    if start_stamps and end_stamps:
        merged_state['time_stamp'] = f"{min(start_stamps)} - {max(end_stamps)}"
    
    # Merge other keys
    for state in world_states:
        for key, value in state.items():
            if key != 'time_stamp':
                if key not in merged_state:
                    merged_state[key] = value
                else:
                    # Concatenate with space if the key already exists
                    merged_state[key] = f"{merged_state[key]} {value}"
    
    return merged_state 