#!/usr/bin/env python3
import json
import argparse
import sys
import yaml
import os

from flask import Flask, request, jsonify, send_from_directory, Response


from chatLLM.gemini import gemini
from chatLLM.openAI import openAI
from ASR_main import create_asr_description
from VLM_main import create_video_description
from utils.index_utils import load_data, group_frame_and_transcription, interactive_chat, process_query_for_web

# Global variables to store processed data
g_audio_chunks = []
g_vision_chunks = []
g_provider_name = ""
g_model = ""

app = Flask(__name__, static_folder="viewer")

@app.route('/')
def index():
    return send_from_directory('viewer', 'video_chat.html')

@app.route('/video/<path:video_path>')
def serve_video(video_path):
    # Get the video directory from the full path
    video_dir = os.path.dirname(video_path)
    video_file = os.path.basename(video_path)
    return send_from_directory(video_dir, video_file)

@app.route('/api/clips')
def get_clips():
    # This is a placeholder - in a real app, you might generate clips dynamically
    # based on video analysis or use stored clip data
    clips = [
        {"label": "Random Clip 1", "start": 120, "end": 180},  # 2:00 - 3:00
        {"label": "Random Clip 2", "start": 300, "end": 360}   # 5:00 - 6:00
    ]
    return jsonify(clips)

@app.route('/api/chat', methods=['POST'])
def chat():
    global g_audio_chunks, g_vision_chunks, g_provider_name, g_model
    
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    if not g_audio_chunks and not g_vision_chunks:
        return jsonify({"error": "No video data loaded. Please process a video first."}), 400
    
    response_data = process_query_for_web(query, g_audio_chunks, g_vision_chunks, g_provider_name, g_model)
    
    # Return JSON response with the response text and timestamps
    return jsonify(response_data)

def query_engine(video_path:str,video_chunk_length:int,vision_cfg:str,audio_cfg:str,use_frame_desc:bool,use_audio_trans:bool,frame_desc_file:str,
                 audio_trans_file:str,model:str,provider:str,web_mode:bool=False):
    global g_audio_chunks, g_vision_chunks, g_provider_name, g_model

    vision_cfg_data = {}
    audio_cfg_data = {}

    with open(vision_cfg, "r") as file:
        vision_cfg_data = yaml.safe_load(file)
    
    with open(audio_cfg, "r") as file:
        audio_cfg_data = yaml.safe_load(file)

    if audio_cfg_data["interval"] != vision_cfg_data["interval"]:
        ValueError("Interval should be same for parsing vision and audio inputs.")
    

    vision_cfg_data["output_file"] = f"{vision_cfg_data['output_dir']}/vision_extract.json"
    audio_cfg_data["output_file"] = f"{audio_cfg_data['output_dir']}/audio_extract.json"
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

    audio_chunks, vision_chunks = group_frame_and_transcription(
        audio_data=audio_data, 
        vision_data=vision_data,
        video_path=video_path, 
        video_chunk_length=video_chunk_length
    )

    # Store in global variables for web mode
    g_audio_chunks = audio_chunks
    g_vision_chunks = vision_chunks
    g_provider_name = provider
    g_model = model
    
    if web_mode:
        # Start web server
        host = '0.0.0.0'
        port = 8080
        print(f"\n=== Starting web server at http://{host}:{port} ===")
        print(f"Navigate to http://localhost:{port}?video_path={video_path} in your browser")
        app.run(host=host, port=port)
    else:
        # Use CLI mode
        interactive_chat(audio_chunks=audio_chunks, vision_chunks=vision_chunks, provider_name=provider, model=model)


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
    parser.add_argument("--web", action="store_true",
                        help="Launch web interface instead of CLI")
    
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
    web_mode = args.web

    query_engine(video_path=video_path,
                video_chunk_length=video_chunk_length,
                vision_cfg=vision_cfg,
                audio_cfg=audio_cfg,
                use_frame_desc=use_frame_desc,
                use_audio_trans=use_audio_trans,
                frame_desc_file=frame_desc_file,
                audio_trans_file=audio_trans_file,
                model=model,
                provider=provider,
                web_mode=web_mode)

def main():
    parser = argparse.ArgumentParser(description="Query solver based on the video.")
    parser.add_argument("--video_cfg_file",type=str, default="./configs/query_engine.yml",
                        help="Config file to be used for processing.")
    parser.add_argument("--web", action="store_true",
                        help="Launch web interface instead of CLI")
    
    args = parser.parse_args()

    video_cfg_file = args.video_cfg_file
    web_mode = args.web

    with open(video_cfg_file, "r") as file:
        video_cfg_data = yaml.safe_load(file)
    
    if web_mode:
        video_cfg_data["web_mode"] = True
    
    query_engine(**video_cfg_data)

if __name__ == "__main__":
    main() 