import torch
import os
import json
import argparse

from utils.audio_utils import extract_audio_from_video, format_transcription_to_json
from modelASR.whisper import whisper



def main():
    parser = argparse.ArgumentParser(description="Transcribe audio from a video file")
    parser.add_argument("--video_path", type=str, help="Path to the video file", default="/root/shubham/videos/RR_batting_10min_2.mp4")
    parser.add_argument("--output", type=str, default="results/video_transcription.json", 
                        help="Path to save the output JSON file")
    parser.add_argument("--interval", type=int, default=3, 
                        help="Time interval in seconds for each transcription segment")
    parser.add_argument("--asr_model", type=str, default="whisper",
                        help="Model to be used for generating transcription")
    parser.add_argument("--diarization_model", type=str, default="pyannote/speaker-diarization-3.1",
                        help="Model to be used for diarization")
    parser.add_argument("--base_model_id", type=str, default="openai/whisper-large-v3",
                        help="Base model for whisper model")
    parser.add_argument("--language", type=str, default="en", help="language to be used for transcription")

    args = parser.parse_args()
    
    video_path = args.video_path
    asr_model = args.asr_model
    diarization_model = args.diarization_model
    base_model_id = args.base_model_id
    language=args.language
    interval = args.interval
    output = args.output
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using video path: {video_path}")
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")
        return
    
    # Create results directory if it doesn't exist
    output_dir = os.path.dirname(output)
    if output_dir:  # If output_dir is not empty
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract audio from video
    audio_data, video_duration = extract_audio_from_video(video_path=video_path)
    
    # Transcribe audio
    
    if asr_model == "whisper":
        model = whisper(diarization_model_name=diarization_model, asr_model_name=base_model_id, device=device, language=language)
    
    transcription_chunks = model.transcribe_batch_audio(audio_data["chunk_files"],audio_data["sampling_rate"])
    
    # Format transcription to JSON
    formatted_json = format_transcription_to_json(transcription_chunks, video_duration, interval)
    
    # Save to file
    with open(output, 'w') as f:
        json.dump(formatted_json, f, indent=2)
    
    print(f"Transcription saved to {args.output}")

if __name__ == "__main__":
    main()