import os
import numpy as np

from transformers import pipeline
from pyannote.audio import Pipeline
import torch



from models_ASR.base_model import base_model 
from utils.audio_utils import load_audio


class whisper(base_model):
    def __init__(self, diarization_model_name:str = "pyannote/speaker-diarization-3.1", 
                 asr_model_name:str = "openai/whisper-large-v3", device:str="cuda", 
                 language:str = "en", hugging_face_token_env="HUGGING_FACE_TOKEN"):
        self.pipeline_asr = pipeline(
            "automatic-speech-recognition",
            model=asr_model_name,
            device=device,
            return_timestamps=True,
            generate_kwargs={"language": language, "task": "transcribe"}
        )
        self.pipeline_diarization = Pipeline.from_pretrained(
            diarization_model_name,
            use_auth_token=os.getenv(hugging_face_token_env),
        ) 
        self.pipeline_diarization.to(torch.device(device))
    
    def transcribe_audio(self, chunk_interval:int, chunk_index:int, audio_path:str, sample_rate:int):
        audio_data = load_audio(audio_path).numpy().astype(np.float32)
        diarization = self.pipeline_diarization(audio_path)
        results = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            start_sample = int(segment.start * sample_rate) + chunk_index*chunk_interval
            end_sample = int(segment.end * sample_rate) + chunk_index*chunk_interval
            segment_audio = audio_data[start_sample:end_sample]
            if len(segment_audio) == 0:
                continue
            transcription = self.pipeline_asr(segment_audio)["text"]
            results.append({
                "speaker": speaker,
                "start": segment.start,
                "end": segment.end,
                "text": transcription
            })
        return results
    
    def transcribe_batch_audio(self, chunk_files, sample_rate:int, chunk_interval:int):
        results = []
        for i,chunk_filepath in enumerate(chunk_files):
            chunk_results = self.transcribe_audio(chunk_interval=chunk_interval, chunk_index=i, audio_path=chunk_filepath, sample_rate=sample_rate)
            results.extend(chunk_results)
        
        return results

        
        
