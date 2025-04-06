import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import os
import json
import logging
import argparse
import numpy as np
import time

# Import from utility modules using relative imports
from utils.video_utils import VideoFrameExtractor, get_video_metadata
from utils.format_text_utils import load_yaml_prompt, create_world_state_history, clean_model_response
from utils.db_utils import create_vector_db, store_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

def load_model(checkpoint_path: str, base_model_id: str = "HuggingFaceTB/SmolVLM-Instruct", device: str = "cuda"):
    # Load processor from original model
    processor = AutoProcessor.from_pretrained(base_model_id)
    if checkpoint_path:
        # Load fine-tuned model from checkpoint
        model = AutoModelForImageTextToText.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        ).to("cuda")
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map=device
        ).to("cuda")    

    # By default, don't resize or crop since we're using original resolution
    processor.image_processor.do_resize = False
    processor.image_processor.do_center_crop = False
    processor.image_processor.do_image_splitting = False
    
    return model, processor

def generate_response(model, processor, vector_db_client, video_path: str, question: str, max_frames: int = 50, batch_size: int = 7, show_stats: bool = False, frames_per_context: int = 1, sample_fps: int = 1, resize_frames: bool = True, target_size: int = 384):
    # Extract frames
    frame_extractor = VideoFrameExtractor(max_frames)
    
    # Use the method to extract frames at specified fps with optional cropping
    frames = frame_extractor.extract_frames_at_fps(
        video_path, 
        sample_fps,
        crop_frames=resize_frames,
        target_size=target_size
    )
    
    if resize_frames:
        logger.info(f"Extracted {len(frames)} frames at {sample_fps} FPS with {target_size}x{target_size} cropping")
        # Update processor configuration for resized images
        processor.image_processor.size = (target_size, target_size)
        processor.image_processor.do_resize = False  # We've already resized
        processor.image_processor.do_center_crop = False  # We've already cropped
        processor.image_processor.do_image_splitting = False
    else:
        logger.info(f"Extracted {len(frames)} frames at {sample_fps} FPS with original resolution")
        processor.image_processor.do_image_splitting = False
    
    # Get video metadata for timing calculations
    video_metadata = get_video_metadata(video_path)
    total_duration = video_metadata["duration_seconds"]
    actual_time_per_frame = total_duration / len(frames) if frames else 0
    
    assistant_model,_ = load_model(checkpoint_path=None, base_model_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct", device="cuda")
    
    # Process frames in batches
    all_responses = []
    all_embeddings = []
    
    # Stats tracking
    total_input_tokens = 0
    total_output_tokens = 0
    batch_processing_stats = []
    
    # Process all frames in context-sized groups
    for i in range(0, len(frames), batch_size * frames_per_context):
        # Calculate batch details
        batch_start = i
        batch_end = min(i + batch_size * frames_per_context, len(frames))
        current_batch = i//(batch_size * frames_per_context) + 1
        total_batches = (len(frames) + batch_size * frames_per_context - 1)//(batch_size * frames_per_context)
        
        # Group frames into contexts
        frame_groups = []
        for j in range(batch_start, batch_end, frames_per_context):
            frame_group = frames[j:min(j+frames_per_context, len(frames))]
            # Only use complete groups if more than one frame per context
            if frames_per_context == 1 or len(frame_group) == frames_per_context:
                frame_groups.append(frame_group)
        
        if not frame_groups:
            continue
            
        # Count groups in this batch
        num_groups = len(frame_groups)
        if frames_per_context > 1:
            logger.info(f"Processing batch {current_batch}/{total_batches} with {num_groups} frame groups ({frames_per_context} frames each)")
        else:
            logger.info(f"Processing batch {current_batch}/{total_batches} with {num_groups} individual frames")
        
        # Create batch of messages
        batch_messages = []
        for frame_group in frame_groups:
            # Calculate the actual duration and time interval for this specific group
            actual_frames_in_group = len(frame_group)
            group_duration = actual_frames_in_group * actual_time_per_frame
            group_interval = actual_time_per_frame
            
            # Dynamically customize prompt for this specific group
            customized_prompt = question
            # Replace key parameters with actual values for this group
            customized_prompt = customized_prompt.replace("sequence of {{FRAMES_COUNT}} images", f"sequence of {actual_frames_in_group} images")
            customized_prompt = customized_prompt.replace("spanning {{TOTAL_DURATION_FORMATTED}}", f"spanning {group_duration:.2f} seconds")
            customized_prompt = customized_prompt.replace("with images spaced approximately every {{TIME_INTERVAL_FORMATTED}}", 
                                                         f"with images spaced approximately every {group_interval:.2f} seconds")
            customized_prompt = customized_prompt.replace("These video frames are part of a {{TOTAL_DURATION_FORMATTED}} sequence", 
                                                         f"These video frames are part of a {group_duration:.2f} seconds sequence")
            
            # Debug log the first prompt in each batch to verify our replacements
            if frame_group == frame_groups[0]:
                logger.info(f"Sample of customized prompt for batch {current_batch}: '{customized_prompt[:100]}...'")
            
            # For each context
            content = [{"type": "text", "text": customized_prompt}]
            # Add all images in this group to the content
            for img in reversed(frame_group):  # Reverse to maintain order when inserting at index 0
                content.insert(0, {"type": "image"})
            
            messages = [{"role": "user", "content": content}]
            batch_messages.append(messages)
        # Process inputs for the batch
        batch_inputs = processor(
            text=[processor.apply_chat_template(messages, add_generation_prompt=True) for messages in batch_messages],
            images=[frame_group for frame_group in frame_groups],
            return_tensors="pt",
            padding=True
        ).to(model.device, dtype=torch.bfloat16)
        
        # Track input tokens
        input_tokens_per_group = batch_inputs.input_ids.size(1)
        batch_input_tokens = input_tokens_per_group * num_groups
        total_input_tokens += batch_input_tokens
        
        # Generate responses for the batch
        start_time = time.time()
        outputs = model.generate(
            **batch_inputs,
            max_new_tokens=256 + (128 if frames_per_context > 1 else 0),  # More tokens for multi-frame
            num_beams=5,
            temperature=0.7,
            do_sample=True,
            use_cache=True
        )
        end_time = time.time()
        
        # Track processing time and output tokens
        batch_processing_time = end_time - start_time
        output_tokens_per_group = outputs.size(1) - batch_inputs.input_ids.size(1)
        batch_output_tokens = output_tokens_per_group * num_groups
        total_output_tokens += batch_output_tokens
        
        # Calculate tokens per image in context
        input_tokens_per_frame = input_tokens_per_group / frames_per_context if frames_per_context > 0 else input_tokens_per_group
        output_tokens_per_frame = output_tokens_per_group / frames_per_context if frames_per_context > 0 else output_tokens_per_group
        
        # Record stats for this batch
        batch_processing_stats.append({
            "batch_num": current_batch,
            "frame_groups": num_groups,
            "frames_per_group": frames_per_context,
            "total_frames": sum(len(group) for group in frame_groups),
            "input_tokens": batch_input_tokens,
            "output_tokens": batch_output_tokens,
            "input_tokens_per_group": input_tokens_per_group,
            "output_tokens_per_group": output_tokens_per_group,
            "input_tokens_per_frame": input_tokens_per_frame,
            "output_tokens_per_frame": output_tokens_per_frame,
            "prompt_length": len(customized_prompt) if frame_groups else 0,
            "processing_time": batch_processing_time
        })

        # Decode generated responses
        batch_responses = processor.batch_decode(outputs, skip_special_tokens=True)
        all_responses.extend(batch_responses)
        
        # Store vectors in the database if vector_db_client is provided
        if vector_db_client is not None:
            try:
                # Collect embeddings for storage
                embeddings = [outputs[idx].tolist() for idx in range(min(batch_size, len(frame_groups)))]
                all_embeddings.extend(embeddings)
                
                # Create metadata for embeddings
                if frames_per_context > 1:
                    # For multi-frame contexts, store group information
                    metadata = [
                        {
                            "group_index": (batch_start + idx * frames_per_context) // frames_per_context,
                            "frame_indices": list(range(batch_start + idx * frames_per_context, 
                                                       min(batch_start + (idx + 1) * frames_per_context, len(frames)))),
                            "response": batch_responses[idx],
                            "timestamp_start": (batch_start + idx * frames_per_context) * 
                                            (get_video_metadata(video_path)["duration_seconds"] / len(frames)),
                            "timestamp_end": min(batch_start + (idx + 1) * frames_per_context, len(frames)) * 
                                           (get_video_metadata(video_path)["duration_seconds"] / len(frames))
                        }
                        for idx in range(min(batch_size, num_groups))
                    ]
                    collection_name = "frameGroupRAG"
                else:
                    # For single frames, store individual frame information
                    metadata = [
                        {
                            "frame_index": batch_start + idx * frames_per_context,
                            "response": batch_responses[idx],
                            "timestamp": (batch_start + idx * frames_per_context) * 
                                       (get_video_metadata(video_path)["duration_seconds"] / len(frames))
                        }
                        for idx in range(min(batch_size, num_groups))
                    ]
                    collection_name = "frameRAG"
                
                # Store vectors in the database
                store_embeddings(
                    client=vector_db_client,
                    collection_name=collection_name,
                    embeddings=embeddings,
                    ids=[(batch_start + idx * frames_per_context) // frames_per_context if frames_per_context > 1 
                          else batch_start + idx * frames_per_context 
                          for idx in range(min(batch_size, num_groups))],
                    metadata=metadata
                )
            except Exception as e:
                logger.warning(f"Failed to store embeddings in vector database: {e}")
    
    # Display token usage statistics if requested
    if show_stats:
        # Get example from first batch if available
        example_batch = batch_processing_stats[0] if batch_processing_stats else None
        
        print("\n" + "="*80)
        mode_str = "MULTI-FRAME CONTEXT" if frames_per_context > 1 else "SINGLE-FRAME"
        print(f"📊 CONTEXT TOKEN METRICS ({mode_str})")
        print("="*80)
        
        # Per-context/frame statistics from a single batch
        if example_batch:
            frames_per_group = example_batch["frames_per_group"]
            
            # Use the detailed per-group and per-frame metrics we collected
            input_tokens_per_group = example_batch["input_tokens_per_group"]
            output_tokens_per_group = example_batch["output_tokens_per_group"]
            total_tokens_per_group = input_tokens_per_group + output_tokens_per_group
            
            input_tokens_per_frame = example_batch["input_tokens_per_frame"]
            output_tokens_per_frame = example_batch["output_tokens_per_frame"]
            prompt_length = example_batch["prompt_length"]
            
            print(f"📏 CONTEXT TOKENS (from first batch):")
            print(f"   - Input tokens per context: {input_tokens_per_group}")
            print(f"   - Output tokens per context: {output_tokens_per_group}")
            print(f"   - Total tokens per context: {total_tokens_per_group}")
            print(f"   - Text prompt length: {prompt_length} characters")
            
            if frames_per_context > 1:
                print(f"\n🖼️ PER-FRAME BREAKDOWN:")
                print(f"   - Frames per context: {frames_per_group}")
                print(f"   - Input tokens per frame: {input_tokens_per_frame:.2f}")
                print(f"   - Output tokens per frame: {output_tokens_per_frame:.2f}")
        
        print("="*80 + "\n")
    
    # Return responses and stats
    return all_responses, {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens, "processing_time": sum(stat["processing_time"] for stat in batch_processing_stats)}

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SmolVLM Video Inference with custom prompts')
    parser.add_argument('--prompt_path', type=str, default="/root/shubham/smolVLM/prompts/smolVLM/caption.yaml",
                       help='Path to the YAML file containing the prompt')
    parser.add_argument('--video_path', type=str, default="/root/shubham/videos/drawing_room.mp4",
                       help='Path to the video file to process')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to the model checkpoint')
    parser.add_argument('--base_model', type=str, default="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                       help='Base model ID')
    parser.add_argument('--batch_size', type=int, default=7,
                       help='Number of frames to process in each batch')
    parser.add_argument('--max_frames', type=int, default=30,
                       help='Maximum number of frames to extract from the video')
    parser.add_argument('--interval_seconds', type=int, default=5,
                       help='Time interval in seconds for grouping frames in world state history')
    parser.add_argument('--preserve_json', action='store_true',
                       help='Preserve JSON structure in world state history')
    parser.add_argument('--use_vector_db', action='store_true',
                       help='Store frame embeddings in Qdrant vector database')
    parser.add_argument('--show_stats', action='store_true',
                       help='Display processing statistics')
    parser.add_argument('--frames_per_context', type=int, default=5,
                       help='Number of frames to include in each context (default: 1, single-frame mode)')
    parser.add_argument('--sample_fps', type=int, default=1,
                       help='Frames per second to sample from the video (default: 1)')
    parser.add_argument('--no_resize_frames', action='store_true',
                       help='Do not resize or crop frames (use original resolution)')
    parser.add_argument('--target_size', type=int, default=384,
                       help='Target size for resizing frames (default: 384)')
    args = parser.parse_args()
    
    # Configuration
    checkpoint_path = args.checkpoint_path
    base_model_id = args.base_model
    video_path = args.video_path
    yaml_prompt_path = args.prompt_path
    batch_size = args.batch_size
    max_frames = args.max_frames
    preserve_json = args.preserve_json
    interval_seconds = args.interval_seconds
    use_vector_db = args.use_vector_db
    show_stats = args.show_stats
    frames_per_context = args.frames_per_context
    sample_fps = args.sample_fps
    resize_frames = not args.no_resize_frames
    target_size = args.target_size
    
    # Statistics tracking
    start_time_total = time.time()
    
    # Check if files exist
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return
    
    if not os.path.exists(yaml_prompt_path):
        logger.error(f"Prompt YAML file not found: {yaml_prompt_path}")
        return

    # Get video metadata
    try:
        video_metadata = get_video_metadata(video_path)
        total_frames = video_metadata["frame_count"]
        fps = video_metadata["fps"]
        total_duration = video_metadata["duration_seconds"]
        
        # Display basic statistics if requested
        if show_stats:
            minutes, seconds = divmod(total_duration, 60)
            hours, minutes = divmod(minutes, 60)
            
            print("\n" + "="*80)
            print("🎬 VIDEO PROCESSING STATISTICS")
            print("="*80)
            print(f"📹 Video path: {video_path}")
            print(f"⏱️ Total duration: {int(hours)}h {int(minutes)}m {int(seconds)}s ({total_duration:.2f} seconds)")
            print(f"🎞️ Total frames in video: {total_frames}")
            print(f"⚡ Video FPS: {fps}")
            
            # Calculate sampling rate
            actual_frames = min(max_frames, total_frames)
            sampling_interval = total_duration / actual_frames if actual_frames > 0 else 0
            
            print(f"🔍 Sampling configuration:")
            print(f"   - Requested max frames: {max_frames}")
            print(f"   - Requested sampling rate: {sample_fps} FPS")
            print(f"   - Sampling interval: {1/sample_fps:.2f} seconds")
            print(f"   - Actual frames to process: {actual_frames}")
            
            if frames_per_context > 1:
                print(f"   - Using multi-frame processing with {frames_per_context} frames per context")
                print(f"   - Total contexts: {actual_frames // frames_per_context + (1 if actual_frames % frames_per_context else 0)}")
            
            # Calculate world state breakdown
            num_intervals = int(np.ceil(total_duration / interval_seconds))
            avg_frames_per_interval = actual_frames / num_intervals if num_intervals > 0 else 0
            
            print(f"\n📝 WORLD STATE ORGANIZATION:")
            print(f"   - Interval length: {interval_seconds} seconds")
            print(f"   - Number of intervals: {num_intervals}")
            print(f"   - Avg. frames per interval: {avg_frames_per_interval:.2f}")
            
            # Hardware utilization
            print(f"\n💻 HARDWARE:")
            print(f"   - Processing device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
            if torch.cuda.is_available():
                print(f"   - GPU: {torch.cuda.get_device_name(0)}")
            
            print(f"\n⚙️ CONFIGURATION:")
            print(f"   - Batch size: {batch_size}")
            print(f"   - Model: {base_model_id}")
            print(f"   - Vector DB enabled: {'Yes' if use_vector_db else 'No'}")
            if resize_frames:
                print(f"   - Frame resizing: {target_size}x{target_size}")
            else:
                print(f"   - Using original frame resolution (no cropping)")
            print("="*80 + "\n")
            
    except ValueError as e:
        logger.error(f"Error getting video metadata: {e}")
        return
    
    # Calculate frame interval based on max_frames
    frames_count = min(max_frames, total_frames)
    time_interval = total_duration / frames_count if frames_count > 0 else 0
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    logger.info("Device: "+  device)
    logger.info("Loading model...")
    model_load_start = time.time()
    model, processor = load_model(checkpoint_path, base_model_id, device)
    model_load_time = time.time() - model_load_start
    
    # Create vector db client if requested
    vector_db_client = None
    if use_vector_db:
        try:
            logger.info("Creating vector database client")
            vector_db_client = create_vector_db()
        except Exception as e:
            logger.error(f"Failed to create vector database: {e}")
            logger.warning("Continuing without vector database")
    
    # Load prompt from YAML
    logger.info(f"Loading prompt from {yaml_prompt_path}")
    
    # Use multi-frame prompt if available
    if frames_per_context > 1 and not os.path.basename(yaml_prompt_path).startswith('multi_frame'):
        # Check if there's a multi_frame.yaml in the same directory
        prompt_dir = os.path.dirname(yaml_prompt_path)
        multi_frame_path = os.path.join(prompt_dir, "multi_frame.yaml")
        if os.path.exists(multi_frame_path):
            logger.info(f"Using multi-frame prompt template from {multi_frame_path}")
            yaml_prompt_path = multi_frame_path
    
    question = load_yaml_prompt(
        yaml_prompt_path, 
        frames_count=frames_count,
        time_interval=time_interval,
        total_duration=total_duration,
        frames_per_context=frames_per_context
    )
    if not question:
        logger.error("Failed to load prompt from YAML file.")
        return
    
    # Generate response
    logger.info("Generating responses...")
    start_time_processing = time.time()
    
    responses, stats = generate_response(
        model=model,
        processor=processor,
        vector_db_client=vector_db_client,
        video_path=video_path,
        question=question,
        max_frames=max_frames,
        batch_size=batch_size,
        show_stats=show_stats,
        frames_per_context=frames_per_context,
        sample_fps=sample_fps,
        resize_frames=resize_frames,
        target_size=target_size
    )
    
    processing_time = time.time() - start_time_processing
    
    if responses:
        logger.info(f"Successfully generated {len(responses)} descriptions")
    else:
        logger.warning("No responses were generated")
        return
    
    # Create world state history
    logger.info("Creating world state history...")
    world_state_start = time.time()
    
    # First clean up the model responses
    cleaned_responses = [clean_model_response(resp) for resp in responses]
    
    # Debug log - compare a sample before and after cleaning
    if responses and len(responses) > 0:
        sample_idx = 0
        logger.info(f"Sample response before cleaning (first 100 chars): '{responses[sample_idx][:100]}...'")
        logger.info(f"Sample response after cleaning (first 100 chars): '{cleaned_responses[sample_idx][:100]}...'")
    
    world_state_history, free_form_text = create_world_state_history(
        video_path, 
        cleaned_responses,  # Use the cleaned responses
        frame_interval_seconds=interval_seconds,
        preserve_json_structure=preserve_json
    )
    world_state_time = time.time() - world_state_start
    
    # Save world state history to file
    video_name = os.path.basename(video_path).split('.')[0]
    os.makedirs("results", exist_ok=True)
    
    # Save as JSON
    with open(f"results/{video_name}_world_state_history.json", "w") as f:
        json.dump(world_state_history, f, indent=2)
    
    # Save free-form text representation
    with open(f"results/{video_name}_world_state_text.txt", "w") as f:
        f.write(free_form_text)
    
    logger.info(f"World state history saved to results/{video_name}_world_state_history.json")
    logger.info(f"Free-form text representation saved to results/{video_name}_world_state_text.txt")
    
    # Final statistics
    if show_stats:
        end_time_total = time.time()
        total_time = end_time_total - start_time_total
        
        print("\n" + "="*80)
        print("🏁 PROCESSING SUMMARY")
        print("="*80)
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Model loading time: {model_load_time:.2f} seconds")
        print(f"Frame processing time: {processing_time:.2f} seconds")
        print(f"World state creation time: {world_state_time:.2f} seconds")
        
        # Performance metrics
        print(f"\n⚡ PERFORMANCE:")
        print(f"   - Real-time factor: {total_duration / processing_time:.2f}x")
        print("="*80)

if __name__ == "__main__":
    with torch.no_grad():
        # Use main() for normal operation
        main()
        

