from google import genai
import time
from google.genai import types
from models_VLM.base_model import base_model
import os,time

class gemini(base_model):
    def __init__(self, base_model: str="gemini-2.0-flash", apikey_var: str = "GEMINI_VISION_API_KEY"):
        self.client = genai.Client(api_key=os.getenv(apikey_var))
        self.base_model_str = base_model



    def  batch_inference(self, frames, question, batch_size, frames_per_context, logger, temperature: float = 0.7, max_tokens: int = 512, min_tokens: int = 512):
        all_responses = []
        
        # Stats tracking
        total_input_tokens = 0
        total_output_tokens = 0
        batch_processing_stats = []
        total_batches = (len(frames) + batch_size * frames_per_context - 1)//(batch_size * frames_per_context)
        
        # Process all frames in context-sized groups
        for i in range(0, len(frames), batch_size * frames_per_context):
            # Calculate batch details
            batch_start = i
            batch_end = min(i + batch_size * frames_per_context, len(frames))
            current_batch = i//(batch_size * frames_per_context) + 1
            
            # Group frames into contexts
            frame_groups = []
            for j in range(batch_start, batch_end, frames_per_context):
                frame_group = frames[j:min(j+frames_per_context, len(frames))]
                frame_groups.append(frame_group)
                
            # Count groups in this batch
            num_groups = len(frame_groups)
            logger.info(f"Processing batch {current_batch}/{total_batches} with {num_groups} frame groups ({frames_per_context} frames each)")

            start_time = time.time()
            batch_input_tokens = 0
            batch_output_tokens = 0
            for frame_group in frame_groups:
                try:
                    response  = self.client.models.generate_content( model = self.base_model_str, 
                                                        contents = frame_group + [question],
                                                        config=types.GenerateContentConfig(
                                                            max_output_tokens=max_tokens,
                                                            temperature=temperature
                                                        ))
                except:
                    time.sleep(30)
                    response  = self.client.models.generate_content( model = self.base_model_str, 
                                                        contents = frame_group + [question],
                                                        config=types.GenerateContentConfig(
                                                            max_output_tokens=max_tokens,
                                                            temperature=temperature
                                                        ))
                metadata = response.usage_metadata
                batch_input_tokens += metadata.prompt_token_count
                batch_output_tokens += metadata.candidates_token_count
                all_responses.append(response.text)
        
            end_time = time.time()
            batch_processing_time  = end_time - start_time
            total_input_tokens += batch_input_tokens
            total_output_tokens += batch_output_tokens

            # Record stats for this batch
            batch_processing_stats.append({
                "batch_num": current_batch,
                "frame_groups": num_groups,
                "frames_per_group": frames_per_context,
                "total_frames": sum(len(group) for group in frame_groups),
                "input_tokens": batch_input_tokens,
                "output_tokens": batch_output_tokens,
                "input_tokens_per_group": batch_input_tokens/num_groups,
                "output_tokens_per_group": batch_output_tokens/num_groups,
                "input_tokens_per_frame": batch_input_tokens/(num_groups*frames_per_context),
                "output_tokens_per_frame": batch_output_tokens/(num_groups*frames_per_context),
                "prompt_length": len(question) if frame_groups else 0,
                "processing_time": batch_processing_time
            })
        return all_responses,batch_processing_stats,total_input_tokens,total_output_tokens


            

        
