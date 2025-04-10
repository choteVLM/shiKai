#!/usr/bin/env python3
import json
import argparse
import sys
import yaml

from chatLLM.gemini import gemini
from chatLLM.openAI import openAI
from ASR_video_inference import create_asr_description
from VLM_video_inference import create_video_description
from utils.index_utils import load_data, group_frame_and_transcription, interactive_chat


def query_engine(video_path:str,video_chunk_length:int,vision_cfg:str,audio_cfg:str,use_frame_desc:bool,use_audio_trans:bool,frame_desc_file:str,
                 audio_trans_file:str,model:str,provider:str):

    vision_cfg_data = {}
    audio_cfg_data = {}

    with open(vision_cfg, "r") as file:
        vision_cfg_data = yaml.safe_load(file)
    
    with open(audio_cfg, "r") as file:
        audio_cfg_data = yaml.safe_load(file)

    if audio_cfg_data["interval"] != vision_cfg_data["interval"]:
        ValueError("Interval should be same for parsing vision and audio inputs.")
    

    vision_cfg_data["output_file"] = f"{vision_cfg_data["output_dir"]}/vision_extract.json"
    audio_cfg_data["output_file"] = f"{audio_cfg_data["output_dir"]}/audio_extract.json"
    vision_cfg_data["video_path"] = video_path
    audio_cfg_data["video_path"] = video_path
    
    vision_cfg_data.pop("output_dir",None)
    audio_cfg_data.pop("output_dir",None)

    print("Configuration for vision context extraction ...")
    for prop, value in vision_cfg_data.items():
        print("     {}: {}".format(prop,value))

    print("")
    print("")
    print("Configuration for audio context extraction ...")
    for prop, value in audio_cfg_data.items():
        print("     {}: {}".format(prop,value))

    if not use_frame_desc:
        print("")
        print("")
        print("STEP1: Vision processing ...")
        create_video_description(**vision_cfg_data)
    else:
        vision_cfg_data["output_file"] = frame_desc_file

    if not use_audio_trans:
        print("")
        print("")    
        print("STEP2: Audio processing ...")
        create_asr_description(**audio_cfg_data)
    else:
        audio_cfg_data["output_file"] = audio_trans_file


    vision_data = load_data(vision_cfg_data["output_file"])

    audio_data = load_data(audio_cfg_data["output_file"])

    audio_chunks,vision_chunks = group_frame_and_transcription(audio_data=audio_data, vision_data=vision_data,
                                                               video_path=video_path, video_chunk_length=video_chunk_length)

    
    interactive_chat(audio_chunks=audio_chunks,vision_chunks=vision_chunks, provider_name=provider,model=model)


def cmd_main():
    parser = argparse.ArgumentParser(description="Query solver based on the video.")
    parser.add_argument("--video_path", "-f", default="results/video.mp4",
                        help="Path to the video")
    parser.add_argument("--video_chunk_length",type=int, default=300,
                        help="chunk length to process by chat model.")
    parser.add_argument("--vision_extract_config", default="configs/video_extract.cfg",
                        help="configuration to be used for vision input context.")
    parser.add_argument("--audio_extract_config", default="configs/audio_extract.cfg",
                        help="configuration to be used for audio input context.")
    parser.add_argument("--use_frame_description", action="store_true",
                        help="Directly use the frame description.")
    parser.add_argument("--frame_description_file",type=str, default="/tmp/frame_desc.json",
                        help="Path to the frame description file")
    parser.add_argument("--use_audio_transcription", action="store_true",
                        help="Directly use the audio transcription.")
    parser.add_argument("--audio_transcription_file",type=str, default="/tmp/audio_desc.json",
                        help="Path to the audio transcription file.")
    parser.add_argument("--provider", "-p", choices=["OpenAI", "Gemini"],
                        help="AI provider to use (OpenAI, Gemini)")
    parser.add_argument("--model", "-m", help="Model to use")
    
    
    args = parser.parse_args()

    video_path = args.video_path
    video_chunk_length = args.video_chunk_length
    vision_cfg = args.vision_extract_config
    audio_cfg = args.audio_extract_config
    use_frame_desc = args.use_frame_description
    use_audio_trans = args.use_audio_transcription
    frame_desc_file = args.frame_description_file
    audio_trans_file = args.audio_transcription_file
    model = args.model
    provider = args.provider

    query_engine(video_path=video_path,video_chunk_length=video_chunk_length,vision_cfg=vision_cfg,audio_cfg=audio_cfg,use_frame_desc=use_frame_desc,
                 use_audio_trans=use_audio_trans,frame_desc_file=frame_desc_file,audio_trans_file=audio_trans_file,model=model,provider=provider)

def main():
    parser = argparse.ArgumentParser(description="Query solver based on the video.")
    parser.add_argument("--video_cfg_file",type=str, default="./configs/query_engine.yml",
                        help="Config file to be used for processing.")
    
    args = parser.parse_args()

    video_cfg_file = args.video_cfg_file

    with open(video_cfg_file, "r") as file:
        video_cfg_data = yaml.safe_load(file)
    
    query_engine(**video_cfg_data)

if __name__ == "__main__":
    main() 