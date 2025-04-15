import json
from typing import List, Dict, Any, Tuple
import sys
import re

from chatLLM.base_model import chatModel
from chatLLM.gemini import gemini
from chatLLM.openAI import openAI

from prompts.video_clip import generic_query_prompt
from prompts.chatbot import synthesis_prompt
from utils.video_utils import get_video_metadata
from utils.format_text_utils import extract_json_response


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load the JSON data from the file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def get_all_available_models() -> Dict[str, List[str]]:
    """Get all available models from all providers."""
    available_models = {}
    
    # Try to get OpenAI models
    openai_api = openAI()
    try:
        openai_api.setup()
        available_models["OpenAI"] = openai_api.list_models()
    except Exception as e:
        print(f"Error setting up OpenAI API: {e}")
        available_models["OpenAI"] = openai_api.OPENAI_PREFERRED_MODELS
    
    # Try to get Gemini models
    gemini_api = gemini()
    try:
        gemini_api.setup()
        available_models["Gemini"] = gemini_api.list_models()
    except Exception as e:
        print(f"Error setting up Gemini API: {e}")
        available_models["Gemini"] = gemini_api.GEMINI_PREFERRED_MODELS
    
    return available_models

def select_provider_and_model() -> Tuple[str, chatModel]:
    """Let the user select a provider and model."""
    providers = {
        "OpenAI": openAI,
        "Gemini": gemini
    }
    
    print("Available AI providers:")
    for i, provider in enumerate(providers.keys(), 1):
        print(f"{i}. {provider}")
    
    choice = input("Select a provider (number): ")
    try:
        provider_index = int(choice) - 1
        provider_name = list(providers.keys())[provider_index]
    except (ValueError, IndexError):
        print("Invalid selection. Using OpenAI as default.")
        provider_name = "OpenAI"
    
    # Create the API instance
    api = providers[provider_name]()
    api.setup()
    
    # Select a model
    api.select_model()
    
    return provider_name, api

def clean_model_response(text):
    """Clean up model response by removing quotes, newlines and extra whitespace."""
    # Remove surrounding quotes (both single and double)
    text = re.sub(r'^["\'](.*)["\']$', r'\1', text)
    
    # Replace escaped newlines
    text = text.replace('\\n', ' ')
    
    # Remove any actual newlines
    text = text.replace('\n', ' ')
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def interactive_chat(audio_chunks, vision_chunks, provider_name: str, model:str):
    if provider_name.lower() == "openai":
        api = openAI()
    elif provider_name.lower() == "gemini":
        api = gemini()
    else:
        print(f"Unknown provider: {provider_name}")
        sys.exit(1)
    api.select_model(model)

    """Run an interactive chat session with the selected model."""
    print("\n" + "="*50)
    print(f"Interactive chat session with {provider_name}")
    print("Type 'exit', 'quit', or 'q' to end the session")
    print("="*50 + "\n")
    
    while True:
        query = input("\nEnter your query about the video: ")
        
        # Check if the user wants to exit
        if query.lower() in ['exit', 'quit', 'q']:
            print("Ending chat session. Goodbye!")
            break
        
        # Query the model
        print(f"\nAnalyzing video to answer your question...")

        responses = []
        for chunk_index in range(0, min(len(audio_chunks),len(vision_chunks))):
            # Prepare the prompt
            prompt = generic_query_prompt.format(query=query, frames=vision_chunks[chunk_index], transcriptions= audio_chunks[chunk_index])
            resp = api.generate(prompt)
            resp = extract_json_response(resp)
            responses.append(resp)
        
        # Parse responses to JSON (silently)
        json_responses = []
        for response in responses:
            try:
                response_json = json.loads(response)
                json_responses.append(response_json)
            except json.JSONDecodeError:
                json_responses.append({"error": "Failed to parse as JSON", "raw_response": response})
        
        # Generate synthesis response
        synthesis_input = "\n".join([json.dumps(resp) for resp in json_responses])
        synthesis_prompt_formatted = synthesis_prompt.format(query=query, responses=synthesis_input)
        synthesis_resp = api.generate(synthesis_prompt_formatted)
        
        # Clean up the response
        synthesis_resp = clean_model_response(synthesis_resp)
        
        # Display only the final response to the user
        print("\n" + synthesis_resp)

def timestamp_str_to_sec(time_stamp:str):
    """Convert timestamp string in HH:MM:SS format to seconds."""
    hours, minutes, seconds = map(int, time_stamp.split(':'))
    return hours * 3600 + minutes * 60 + seconds

def group_frame_and_transcription(audio_data, vision_data, video_path, video_chunk_length): ##TODO: variations here
    audio_chunks = []
    vision_chunks = []
    audio_index = 0
    vision_index = 0
    video_metadata = get_video_metadata(video_path=video_path)
    current_time = 0
    while current_time < video_metadata['duration_seconds']:
        current_end_time = current_time + video_chunk_length
        audio_desc = ""
        while audio_index < len(audio_data):
            start_time = audio_data[audio_index]["time_stamp"].split(" - ")[0].strip()
            if timestamp_str_to_sec(start_time) < current_end_time:
                audio_desc = "{}\n{}".format(audio_desc,audio_data[audio_index])
            else:
                break
            audio_index = audio_index + 1
        if audio_desc != "":
            audio_chunks.append(audio_desc)

        video_desc = ""
        while vision_index < len(vision_data):
            start_time = vision_data[vision_index]["time_stamp"].split(" - ")[0].strip()
            if timestamp_str_to_sec(start_time) < current_end_time:
                video_desc = "{}\n{}".format(video_desc,vision_data[vision_index])
            else:
                break
            vision_index = vision_index + 1
        if video_desc != "":
            vision_chunks.append(video_desc)
    
        current_time = current_end_time
    return audio_chunks,vision_chunks

def process_query_for_web(query, audio_chunks, vision_chunks, provider_name, model):
    """Process a query for web interface and return the response."""
    if provider_name.lower() == "openai":
        api = openAI()
    elif provider_name.lower() == "gemini":
        api = gemini()
    else:
        return {"error": f"Unknown provider: {provider_name}"}
    
    api.select_model(model)
    
    # Process individual segments silently
    segment_responses = []
    raw_timestamps = []  # Collect raw timestamps before aggregation
    
    for chunk_index in range(0, min(len(audio_chunks), len(vision_chunks))):
        prompt = generic_query_prompt.format(query=query, frames=vision_chunks[chunk_index], transcriptions=audio_chunks[chunk_index])
        resp = api.generate(prompt)
        resp = extract_json_response(resp)
        try:
            response_json = json.loads(resp)
            segment_responses.append(response_json)
            
            # Extract timestamps from segment response
            if isinstance(response_json, dict) and "relevant_scenes" in response_json:
                for scene in response_json["relevant_scenes"]:
                    if "time_stamp" in scene:
                        time_range = scene["time_stamp"]
                        if " - " in time_range:
                            start_time, end_time = time_range.split(" - ")
                            start_sec = timestamp_str_to_sec(start_time)
                            end_sec = timestamp_str_to_sec(end_time)
                            raw_timestamps.append({
                                "start": start_sec,
                                "end": end_sec,
                                "label": scene.get("description", "Relevant Scene")
                            })
        except json.JSONDecodeError:
            segment_responses.append({"raw_response": resp, "error": "Failed to parse as JSON"})
    
    # Generate synthesis response
    synthesis_input = "\n".join([json.dumps(resp) for resp in segment_responses])
    synthesis_prompt_formatted = synthesis_prompt.format(query=query, responses=synthesis_input)
    synthesis_resp = api.generate(synthesis_prompt_formatted)
    
    # Clean up the response
    synthesis_resp = clean_model_response(synthesis_resp)
    
    # Aggregate the timestamps efficiently
    final_timestamps = []
    
    if raw_timestamps:
        # Sort timestamps by start time
        raw_timestamps.sort(key=lambda x: x["start"])
        
        # Initialize with the first timestamp
        current_group = raw_timestamps[0].copy()
        
        # Process remaining timestamps, merging when appropriate
        for ts in raw_timestamps[1:]:
            # If the current timestamp starts within 2s of the previous group ending
            if ts["start"] - current_group["end"] <= 2:
                # Extend the end time if needed
                current_group["end"] = max(current_group["end"], ts["end"])
                # Combine labels if different
                if ts["label"] != current_group["label"]:
                    current_group["label"] = f"{current_group['label']}; {ts['label']}"
            else:
                # Current timestamp doesn't overlap, save the current group and start a new one
                final_timestamps.append(current_group)
                current_group = ts.copy()
        
        # Don't forget to add the last group
        final_timestamps.append(current_group)
    
    # Return both the response and extracted timestamps
    return {
        "response": synthesis_resp,
        "timestamps": final_timestamps
    }

    
