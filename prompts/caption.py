caption_prompt = """
  MAIN INSTRUCTIONS:
  
  Your task is to analyze video frames extracted from a {TOTAL_DURATION_FORMATTED} long video for a detailed 
  video understanding exercise. 

  I will provide a sequence of {FRAMES_COUNT} images spanning {TOTAL_DURATION_FORMATTED}, with images spaced approximately every {TIME_INTERVAL_FORMATTED}.
    
  Examine the video frames closely and generate a comprehensive caption by strictly following the steps below:
  

  Step 1: **Scene Context**: Observe the video. What is the primary setting and activity in the video?


  Step 2: **Motion Description**: Identify and describe any significant motion or actions taking place.
  

  Step 3: **Spatial Relationship Analysis**: Examine and report on the spatial relationships between key objects or characters in the video frames. 
  Describe the positioning and orientation of each element relative to others.
  

  Step 4: **Detailed Object Analysis**: List the key objects and characters in the frame. Describe their color, shape, texture, and 
  any other notable features with precision. Focus on specific details like clothing, accessories, and colors.
  

  Step 5: **Temporal Relationship Context**: These video frames are part of a {TOTAL_DURATION_FORMATTED} sequence, therefore explain any temporal relationships.
  

  Step 6: **Additional Details**: Note any other important details or elements that stand out but are not covered by the above points, 
  i.e.: gender, hair color, colors of accessories and other attributes in the video frames.
  

  Step 7: **Summary**: Provide a concise yet comprehensive summary capturing the key elements and takeaways from this {TOTAL_DURATION_FORMATTED} video following Steps 1 to 6 above. 
  Your caption should encapsulate the scene's key aspects, offering a comprehensive understanding of its environment, activities and context.


  GUIDELINES:

  1. Strictly return your results in JSON format. Please see the example below:
  ```json
  {{
    "Scene Context": "A busy beach scene with families and surfers enjoying the sunny day.",
    "Motion Description": "Children are building a sandcastle, a dog is running towards the water, and a surfer is catching a wave.",
    "Spatial Relationship Analysis": "The sandcastle is in the foreground, the dog approaches from the left, and the surfer moves from right to center.",
    "Detailed Object Analysis": "Children are wearing colorful swimwear; the dog is a golden retriever; the surfer is wearing a blue and white wetsuit.",
    "Temporal Relationship Context": "The progression from building sandcastles to engaging with the incoming tide indicates a passage of time towards later afternoon.",
    "Additional Details": "One child has red hair; the dog's leash is lying abandoned on the sand; multiple surfboards are visible in the background.",
    "Summary": "The video frames depict a joyful beach day emphasizing family activities, interaction with nature, and surfing as a key activity, showcasing the beach's vibrant atmosphere."
  }}
  ```

  2. VERY IMPORTANT: YOU ARE ALLOWED TO USE A MAXIMUM OF 200 words in total
"""



  