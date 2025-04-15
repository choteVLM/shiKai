import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from models_VLM.base_model import base_model
import time

class smolVLM(base_model):
    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = "HuggingFaceTB/SmolVLM-Instruct",
        device: str = "cuda",
        target_size: int = 384
    ):
        self.processor = AutoProcessor.from_pretrained(model_name)
        if checkpoint_path:
            # Load fine-tuned model from checkpoint
            self.model = AutoModelForImageTextToText.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map=device
            ).to(device)
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device
            ).to(device)    

        # Configure processor for video frames
        self.processor.image_processor.size = (target_size, target_size)
        self.processor.image_processor.do_resize = False
        self.processor.image_processor.do_center_crop = False
        self.processor.image_processor.do_image_splitting = False

    def _formulate_batch_inputs(self, question, frame_groups):
        batch_messages = []
        for frame_group in frame_groups:
            # For each context
            content = [{"type": "image"}]*len(frame_group) + [{"type": "text", "text": question}]
            messages = [{"role": "user", "content": content}]
            batch_messages.append(messages)
        return batch_messages
    
    
    def batch_inference(self, frames, question, batch_size, frames_per_context, logger, temperature: float = 0.7, max_token = 256, min_token = 256):
        ## Remove 
        # assistant_model,_ = load_model(checkpoint_path=None, base_model_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct", device="cuda")
        
        # Process frames in batches
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

            
            # Create batch of messages
            batch_messages = self._formulate_batch_inputs(question=question, frame_groups=frame_groups)
            
            # Process inputs for the batch
            batch_inputs = self.processor(
                text=[self.processor.apply_chat_template(messages, add_generation_prompt=True) for messages in batch_messages],
                images=[frame_group for frame_group in frame_groups],
                return_tensors="pt",
                padding=True
            ).to(self.model.device, dtype=torch.bfloat16)
            
            # Track input tokens
            input_tokens_per_group = batch_inputs.input_ids.size(1)
            batch_input_tokens = input_tokens_per_group * num_groups
            total_input_tokens += batch_input_tokens
            
            # Generate responses for the batch
            start_time = time.time()
            outputs = self.model.generate(
                **batch_inputs,
                # Make embedding in the same space
                max_new_tokens=max_token,
                min_new_tokens=min_token, 
                num_beams=5,
                temperature=temperature,
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
                "prompt_length": len(question) if frame_groups else 0,
                "processing_time": batch_processing_time
            })

            # Decode generated responses
            batch_responses = self.processor.batch_decode(outputs, skip_special_tokens=True)
            all_responses.extend(batch_responses)

        return all_responses, batch_processing_stats, total_input_tokens, total_output_tokens


