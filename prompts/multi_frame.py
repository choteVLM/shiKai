multi_frame_prompt = """
  MAIN INSTRUCTIONS:
  
  Your task is to analyze a sequence of {FRAMES_PER_CONTEXT} consecutive video frames 
  extracted from a {TOTAL_DURATION_FORMATTED} long video.

  I will provide you with a group of {FRAMES_PER_CONTEXT} sequential images spanning 
  approximately {TIME_INTERVAL_FORMATTED} of video time. These frames are part of a longer
  video with {FRAMES_COUNT} total frames sampled at regular intervals.
    
  Examine these consecutive frames carefully and provide analysis showing how the scene 
  changes across this short time sequence. Focus on:
  
  1. **Motion Analysis**: Describe any movement or action occurring across the sequence.
     Track specific objects or subjects as they move through the frames.
  
  2. **Temporal Changes**: Identify what changes from the first frame to the last in this sequence.
     Note any progression of events, activities, or environmental changes.
  
  3. **Causal Relationships**: If applicable, describe cause-and-effect relationships visible
     in this short sequence (e.g., "The person drops the ball, which then bounces").
  
  4. **Consistent Elements**: Identify objects, people, or background elements that remain
     constant throughout this sequence of frames.
  
  5. **Visual Storytelling**: Construct a micro-narrative that captures what's happening
     in this specific segment of the video.
  
  IMPORTANT: Your analysis should focus specifically on the temporal relationship between
  these consecutive frames, treating them as a continuous sequence rather than isolated images.
  Focus on changes and motion instead of static description.
  
  RESPONSE FORMAT:
  ```json
  {{
    "Sequence Timeframe": "Brief statement about where this sequence likely fits within the overall video",
    "Motion Analysis": "Detailed description of movement and action across frames",
    "Key Changes": "List of specific changes observed from first to last frame",
    "Stable Elements": "Description of what remains consistent across all frames",
    "Micro-Narrative": "A short 2-3 sentence story describing what happens in this sequence"
  }}
  ```
  
  Remember that these {FRAMES_PER_CONTEXT} frames represent only {TIME_INTERVAL_FORMATTED} from a
  {TOTAL_DURATION_FORMATTED} video, so focus your analysis on what can be observed within
  just this specific sequence. 
  """