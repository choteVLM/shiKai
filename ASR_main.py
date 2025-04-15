import torch
import os
import json
import argparse

from utils.audio_utils import extract_audio_from_video, format_transcription_to_json
from models_ASR.whisper import whisper

def create_asr_description(video_path:str, asr_model:str, diarization_model:str, base_model_id:str, language: str, interval: int, output_file: str,
                           sampling_rate:int, chunk_interval:int, overlap_sec: int, chunk_dir: str, chunk_file_prefix: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using video path: {video_path}")
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")
        return
    
    # Create results directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:  # If output_dir is not empty
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract audio from video
    audio_data, video_duration = extract_audio_from_video(video_path=video_path,sampling_rate=sampling_rate,chunk_interval=chunk_interval, overlap=overlap_sec,
                                                          chunk_dir=chunk_dir,chunk_file_prefix=chunk_file_prefix)
    
    # Transcribe audio
    
    if asr_model == "whisper":
        model = whisper(diarization_model_name=diarization_model, asr_model_name=base_model_id, device=device, language=language)
    
    transcription_chunks = model.transcribe_batch_audio(audio_data["chunk_files"],audio_data["sampling_rate"])
    
    # Format transcription to JSON
    formatted_json = format_transcription_to_json(transcription_chunks, video_duration, interval)
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(formatted_json, f, indent=2)
    
    print(f"Transcription saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio from a video file")
    parser.add_argument("--video_path", type=str, help="Path to the video file", default="/root/shubham/videos/RR_batting_10min_2.mp4")
    parser.add_argument("--output_file", type=str, default="results/video_transcription.json", 
                        help="Path to save the output JSON file")
    parser.add_argument("--interval", type=int, default=3, 
                        help="Time interval in seconds for each transcription segment")
    parser.add_argument("--asr_model", type=str, default="whisper",
                        help="Model to be used for generating transcription")
    parser.add_argument("--diarization_model", type=str, default="pyannote/speaker-diarization-3.1",
                        help="Model to be used for diarization")
    parser.add_argument("--base_model_id", type=str, default="openai/whisper-large-v3",
                        help="Base model for whisper model")
    parser.add_argument("--language", type=str, default="en", help="Language to be used for transcription")
    parser.add_argument("--sampling_rate", type=int, default=16000,help="Sampling rate for extracting audio")
    parser.add_argument("--chunk_interval",type=int, default=1200, help="Interval to be used for audio chunking")
    parser.add_argument("--overlap_sec", type=int, default=5, help="Overlapping interval for associating transcription to a timestamp.")
    parser.add_argument("--chunk_dir",type=str,default="/tmp/",help="Directory to be used for intermediate results.")
    parser.add_argument("--chunk_file_prefix",type=str,default="chunk_",help="Prefix for file name used to store audio chunks.")


    args = parser.parse_args()
    
    video_path = args.video_path
    asr_model = args.asr_model
    diarization_model = args.diarization_model
    base_model_id = args.base_model_id
    language=args.language
    interval = args.interval
    output_file = args.output_file
    sampling_rate = args.sampling_rate
    chunk_interval = args.chunk_interval
    overlap_sec = args.overlap_sec
    chunk_dir = args.chunk_dir
    chunk_file_prefix = args.chunk_file_prefix
    
    create_asr_description(video_path=video_path, asr_model=asr_model, diarization_model=diarization_model, base_model_id=base_model_id, 
                           language=language, interval=interval, output_file=output_file,sampling_rate=sampling_rate,chunk_interval=chunk_interval,overlap_sec=overlap_sec,
                           chunk_dir=chunk_dir,chunk_file_prefix=chunk_file_prefix)

if __name__ == "__main__":
    main()