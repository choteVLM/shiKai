generic_query_prompt = """
    You are video analyzer and you have been given timestamped frame description and audio transcription below.
    Your Job is to solve the query mentioned in the <query> tag and return the relevant timestamp from the clip.
    You should use the frame description and audio description to answer the query.
    <query>
    {query}
    </query>

    Timestampped Frame Description:
    <timestamped_frames>
    {frames}
    </timestamped_frames>

    Timestampped Audio Transcription:
    <timestamped_transcription>
    {transcriptions}
    </timestamped_transcription>

    Please output your answer as JSON object
    Output Format:
    {{
        "relevant_scenes": [
            {{
                "time_stamp": "timestamp from the video",
                "description": "brief description of what happens in this scene",
                "relevance": "why this scene is relevant to the query"
            }},
            ...
        ],
    }}
    
"""