import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import cv2
import numpy as np
from typing import List
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

class VideoFrameExtractor:
    def __init__(self, max_frames: int = 50):
        self.max_frames = max_frames
        
    def resize_and_center_crop(self, image: Image.Image, target_size: int) -> Image.Image:
        # Get current dimensions
        width, height = image.size
        
        # Calculate new dimensions keeping aspect ratio
        if width < height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))
            
        # Resize
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Center crop
        left = (new_width - target_size) // 2
        top = (new_height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        
        return image.crop((left, top, right, bottom))
        
    def extract_frames(self, video_path: str) -> List[Image.Image]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Calculate frame indices to extract (1fps)
        frame_indices = list(range(0, total_frames, fps))
        
        # If we have more frames than max_frames, sample evenly
        if len(frame_indices) > self.max_frames:
            indices = np.linspace(0, len(frame_indices) - 1, self.max_frames, dtype=int)
            frame_indices = [frame_indices[i] for i in indices]
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame)
                pil_image = self.resize_and_center_crop(pil_image, 384)
                frames.append(pil_image)
        
        cap.release()
        return frames

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

    # Configure processor for video frames
    processor.image_processor.size = (384, 384)
    processor.image_processor.do_resize = False
    processor.image_processor.do_image_splitting = False
    
    return model, processor

def generate_response(model, processor, vector_db_client, video_path: str, question: str, max_frames: int = 50, batch_size: int = 4):
    # Extract frames
    frame_extractor = VideoFrameExtractor(max_frames)
    frames = frame_extractor.extract_frames(video_path)
    logger.info(f"Extracted {len(frames)} frames from video")
    # file = open("FrameDescription.txt", "w")

    assistant_model,_ = load_model(checkpoint_path=None, base_model_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct", device="cuda")
    
    # Process frames in batches
    all_responses = []
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(frames) + batch_size - 1)//batch_size} with {len(batch_frames)} frames")
        
        # Create batch of messages
        batch_messages = []
        for img in batch_frames:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe a detail description of the image. Summarize the image in a few sentences."} 
                    ]
                }
            ]
            batch_messages.append(messages)
        
        # Process inputs for the batch
        batch_inputs = processor(
            text = [processor.apply_chat_template(messages, add_generation_prompt=True) for messages in batch_messages],
            images= [[img] for img in batch_frames],
            return_tensors="pt",
            padding=True
        ).to(model.device, dtype=torch.bfloat16)
        breakpoint()
        
        # Generate responses for the batch
        outputs = model.generate(
            **batch_inputs,
            max_new_tokens=256,
            num_beams=5,
            temperature=0.7,
            #assistant_model=assistant_model,
            do_sample=True,
            use_cache=True
        ).tolist()

        vector_db_client.upsert(collection_name="frameRAG",
                                points = [
                                    PointStruct(id=i + idx, vector=outputs[idx])
                                    for idx in range(0,batch_size)
                                ])
        # batch_responses = processor.batch_decode(outputs, skip_special_tokens=True)
        # all_responses.extend(batch_responses)
        
        # # Write responses to file
        # for j, response in enumerate(batch_responses):
        #     frame_idx = i + j + 1
        #     file.write(f"Frame {frame_idx}:\n {response}\n\n")
    
    # file.close()
    return all_responses


def create_vector_db():
    client = QdrantClient(host="localhost", port=6333)
    if(client.collection_exists(collection_name="frameRAG")):
        client.delete_collection(collection_name="frameRAG")
    client.create_collection(collection_name="frameRAG",
                            vectors_config=VectorParams(size=512,distance=Distance.COSINE))
    return client



def main():
    # Configuration
    #checkpoint_path = "/path/to/your/checkpoint"
    checkpoint_path = None
    base_model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"  
    video_path = "C:/Users/jnama/Downloads/sampled_frames.mp4"
    question = "Describe the video frame by frame."
    batch_size = 4  # Number of frames to process in each batch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    # Load model
    logger.info("Device: "+  device)
    logger.info("Loading model...")
    model, processor = load_model(checkpoint_path, base_model_id, device)
    db_client = create_vector_db()
    
    # Generate response
    logger.info("Generating response...")
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # try:
    responses = generate_response(model, processor, db_client, video_path, question, 60, batch_size)
    # Print results
    print("Question:", question)
    print(f"Generated {len(responses)} frame descriptions")
    print("First frame description:", responses[0] if responses else "No responses generated")
    # except Exception as e:
    #     print(torch.cuda.memory_summary(device=None, abbreviated=False))
    #     print(e)

if __name__ == "__main__":
    with torch.no_grad():
        main()

