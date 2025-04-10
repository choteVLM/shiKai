import os
import json
from datetime import timedelta
import torchaudio
import ffmpeg

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

def load_audio(audio_path:str):
    audio_data, _ = torchaudio.load(audio_path)
    return audio_data.squeeze()


def extract_audio_from_video(video_path, sampling_rate=16000,  chunk_interval:int = 1200, overlap:int = 5, chunk_dir:str = "/tmp/", chunk_file_prefix:str = "chunk_"):
    """Extract audio from video file using ffmpeg"""
    print(f"Extracting audio from {video_path}...")
    
    # Create a temporary audio file
    temp_audio_path = "temp_audio.wav"

    # Use ffmpeg to extract audio from video
    ffmpeg.input(video_path).output(temp_audio_path,
                                    acodec="pcm_s16le",
                                    ar=sampling_rate,
                                    ac=1,
                                    loglevel='error').run(overwrite_output=True)

    
    video_duration  = float(ffmpeg.probe(video_path)['format']['duration'])

    # Load the audio file
    audio_data, _ = torchaudio.load(temp_audio_path)
    audio = audio_data.squeeze()
    sample_len = len(audio)


    chunk_files = []
    boundary = 0
    index = 0
    while boundary < sample_len - 1:
        start_time = max(0, boundary - overlap * sampling_rate)
        end_time = min(sample_len - 1, boundary + chunk_interval * sampling_rate)
        out_path = os.path.join(chunk_dir, f"{chunk_file_prefix}_{index}.wav")

        torchaudio.save(out_path, audio_data[:,start_time:end_time], sampling_rate)
        chunk_files.append(out_path)
        boundary = boundary + chunk_interval*sampling_rate
        index = index + 1
    
    # Display audio statistics
    print("\n=== Audio Statistics ===")
    print(f"Duration: {video_duration:.2f} seconds ({format_timestamp(video_duration)})")
    print(f"Sample Rate: {sampling_rate} Hz")
    print(f"Number of Samples: {len(audio)}")
    print(f"Audio Data Type: {audio.dtype}")
    print(f"Memory Size: {audio.nbytes / (1024 * 1024):.2f} MB")
    print("=======================\n")
    
    # Clean up temporary file
    try:
        os.remove(temp_audio_path)
    except:
        pass
    
    return {"chunk_files": chunk_files, "sampling_rate": sampling_rate}, video_duration

def format_transcription_to_json(transcription_chunks, video_duration, interval:int = 3, threshold:int = 1):
    """Format transcription into the same structure as RR_batting_10min_world_state_history.json"""
    result = []
    # # Process intervals from 0 to video_duration in steps of 'interval' seconds
    current_time = 0
    index = 0
    while current_time < video_duration:
    #     # Find any transcriptions that overlap with this time window
        current_end = current_time + interval
        transcribe = ""
        overwrite_index = index
        while index < len(transcription_chunks):
            chunk_start = transcription_chunks[index]['start']
            chunk_end = transcription_chunks[index]['end']
            if chunk_start > current_end + threshold:
                break
            elif chunk_start <= current_end - threshold:
                overwrite_index = index
            
            if chunk_end > current_time - threshold:
                transcribe = "{}\n{}: {}".format(transcribe, transcription_chunks[index]['speaker'], transcription_chunks[index]['text'])  
            index = index + 1
        
        index = overwrite_index
        if transcribe != "":
            scene_description = {
                "Sequence Timeframe": f"{interval} seconds from a {int(video_duration // 60)} minutes and {int(video_duration % 60)} seconds video",
                "Audio Transcription": transcribe
            }
            
            result.append({
                "time_stamp": create_time_range(current_time, interval),
                "scene_description": json.dumps(scene_description, indent=2)
            })
        
        current_time = current_time + interval
    
    return result