import torch
import os
import json
import argparse
import subprocess
import numpy as np
from pathlib import Path
from transformers import pipeline
from datetime import timedelta
import soundfile as sf

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def create_time_range(start_time, duration=3):
    """Create a time range string in the format 'HH:MM:SS - HH:MM:SS'"""
    end_time = start_time + duration
    return f"{format_timestamp(start_time)} - {format_timestamp(end_time)}"

def extract_audio_from_video(video_path):
    """Extract audio from video file using ffmpeg"""
    print(f"Extracting audio from {video_path}...")
    
    # Create a temporary audio file
    temp_audio_path = "temp_audio.wav"
    
    # Use ffmpeg to extract audio from video
    cmd = [
        "ffmpeg", "-i", video_path, 
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # PCM format
        "-ar", "16000",  # 16kHz sample rate
        "-ac", "1",  # Mono
        "-y",  # Overwrite output file
        temp_audio_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        raise
    
    # Get video duration using ffprobe
    duration_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    
    try:
        result = subprocess.run(duration_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        video_duration = float(result.stdout.decode().strip())
    except subprocess.CalledProcessError as e:
        print(f"Error getting video duration: {e}")
        video_duration = 0
    
    # Load the audio file
    audio_data, sample_rate = sf.read(temp_audio_path)
    
    # Display audio statistics
    print("\n=== Audio Statistics ===")
    print(f"Duration: {video_duration:.2f} seconds ({format_timestamp(video_duration)})")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Number of Samples: {len(audio_data)}")
    print(f"Audio Data Type: {audio_data.dtype}")
    print(f"Memory Size: {audio_data.nbytes / (1024 * 1024):.2f} MB")
    print("=======================\n")
    
    # Clean up temporary file
    try:
        os.remove(temp_audio_path)
    except:
        pass
    
    return {"array": audio_data, "sampling_rate": sample_rate}, video_duration

def transcribe_audio(audio_data, chunk_duration=30):
    """Transcribe audio using Whisper model with timestamps"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Calculate number of chunks
    audio_array = audio_data["array"]
    sample_rate = audio_data["sampling_rate"]
    total_duration = len(audio_array) / sample_rate
    num_chunks = int(np.ceil(total_duration / chunk_duration))
    
    # Display processing configuration
    print("\n=== Whisper Processing Configuration ===")
    print(f"Model: openai/whisper-tiny")
    print(f"Chunk Duration: {chunk_duration} seconds")
    print(f"Approximate Number of Chunks: {num_chunks}")
    print(f"Batch Size: 8")
    print(f"Forced Language: English")
    print(f"Device: {device}")
    print("=====================================\n")
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        chunk_length_s=chunk_duration,
        device=device,
        return_timestamps=True,
        generate_kwargs={"language": "en", "task": "transcribe"}
    )
    
    print("Transcribing audio...")
    print(f"Total audio duration: {total_duration:.2f} seconds")
    
    # Add progress tracking
    result = pipe(audio_data, batch_size=8)
    
    # Display transcription statistics
    print("\n=== Transcription Results ===")
    print(f"Total Chunks Generated: {len(result['chunks'])}")
    print("=============================\n")
    
    return result["chunks"]

def format_transcription_to_json(transcription_chunks, video_duration, interval=3):
    """Format transcription into the same structure as RR_batting_10min_world_state_history.json"""
    result = []
    
    # Process intervals from 0 to video_duration in steps of 'interval' seconds
    current_time = 0
    while current_time < video_duration:
        # Find any transcriptions that overlap with this time window
        current_end = current_time + interval
        relevant_transcriptions = []
        
        for chunk in transcription_chunks:
            chunk_start, chunk_end = chunk["timestamp"]
            
            # Skip chunks with incomplete timestamp information
            if chunk_end is None:
                continue
                
            # Check if this chunk overlaps with the current time window
            if (chunk_start <= current_end and chunk_end >= current_time):
                relevant_transcriptions.append(chunk["text"])
        
        # Create a scene description for this time window
        if relevant_transcriptions:
            text = " ".join(relevant_transcriptions)
            scene_description = {
                "Sequence Timeframe": f"{interval} seconds from a {int(video_duration // 60)} minutes and {int(video_duration % 60)} seconds video",
                "Audio Transcription": text
            }
            
            result.append({
                "time_stamp": create_time_range(current_time, interval),
                "scene_description": json.dumps(scene_description, indent=2)
            })
        
        current_time += interval
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio from a video file")
    parser.add_argument("--video_path", type=str, help="Path to the video file", default="/root/shubham/videos/RR_batting_10min_2.mp4")
    parser.add_argument("--output", type=str, default="results/video_transcription.json", 
                        help="Path to save the output JSON file")
    parser.add_argument("--interval", type=int, default=3, 
                        help="Time interval in seconds for each transcription segment")
    args = parser.parse_args()
    
    video_path = args.video_path
    print(f"Using video path: {video_path}")
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")
        return
    
    # Create results directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:  # If output_dir is not empty
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract audio from video
    audio_data, video_duration = extract_audio_from_video(video_path)
    
    # Transcribe audio
    transcription_chunks = transcribe_audio(audio_data)
    
    # Format transcription to JSON
    formatted_json = format_transcription_to_json(transcription_chunks, video_duration, args.interval)
    
    # Save to file
    with open(args.output, 'w') as f:
        json.dump(formatted_json, f, indent=2)
    
    print(f"Transcription saved to {args.output}")

if __name__ == "__main__":
    main()