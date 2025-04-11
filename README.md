<p align="center">
  <h1 align="center"><b><i><u>Shikai</u></i></b>: Flexible search engine for videos</h1>
</p>
<p align="center">
  <img src="assets/title.png" alt="ShiKai Logo">
</p>



An python CLI tool for video analysis that combines visual language model inference with audio transcription. This tool enables users to query video content and receive relevant timestamps, generating comprehensive descriptions and temporal summaries of both visual and audio elements.

## Features

### Frame Extraction Features
- Extract frames from videos at configurable intervals
- Process frames with visual language models. You can use any VLM by implementing a simple interface
- Generate detailed frame descriptions
- Multi-frame context processing for sequential frame analysis
- Detailed statistics tracking for token usage and performance

### Audio Extraction Features
- Extract audio from the video at a configurable sampling rate
- Process the audio to generate a transcription. You can use any Diarization-ASR pipeline by implementing a simple interface
- Generate detailed transciption with different speaker annotation.

### Query Engine
- Answers user's query by using the Frame description and Audio transcription.
- Plug and play any large language model to answer your query.

## Installation

### Option 1: Install from source (Development Mode)

```bash
git clone https://github.com/choteVLM/shiKai.git
cd shiKai
pip install -e .
```

### Option 2: Install Core Dependencies Only

If you want to skip installing the package's dependencies and only install core requirements:

```bash
pip install -e . --no-deps
pip install -r shiKai/essential-requirements.txt
```

## Usage

```bash
python shiKai/mainIndex.py --video_cfg_file ./configs/query_engine.yml
```

### Config Properties for Query Engine(query_engine.yml)

- video_path: path of the video to be analyzed
- video_chunk_length: video chunks length in seconds whose transcription and frame description can fit in query engine context
- vision_cfg: path of the config file containing configuration for frame description VLM
- audio_cfg: path of the config file containing configuration for audio transcription pipeline
- use_frame_desc: boolean(True/False) representing whether you want to directly use the frame description file without invoking VLM  
- frame_desc_file: path to the frame description file
- use_audio_trans: boolean(True/False) representing whether you want to directly use the audio transcription file without invoking ASR pipeline  
- audio_trans_file: path to the audio transcription file
- provider: LLM provider used for answering your query (like Gemini/OpenAI)
- model: LLM model to be used for answering your query (like gemini-2.0-flash)

```
#### Configuration Example ####

    video_path: ./results/video.mp4
    video_chunk_length: 300
    vision_cfg: ./configs/vision_extract.yml
    audio_cfg: ./configs/audio_extract.yml
    use_frame_desc: False
    frame_desc_file: /tmp/frame_desc.json
    use_audio_trans: False
    audio_trans_file: /tmp/audio_desc.json
    provider: Gemini
    model: gemini-2.0-flash
```

### Config Properties for Frame Description(vision_cfg)

- base_model_id: hugging face model to be used for extracting frame description (like HuggingFaceTB/SmolVLM2-2.2B-Instruct)
- batch_size: batch size to be used for batch inferencing
- max_frames: maximum number of frames to be sampled
- preserve_json: boolean(True/False) used for storing frame description in JSON format
- interval: interval in sec to process together for generating multi frame description
- show_stats: boolean(True/False) used for showing processing stats
- frames_per_context: Nnumbers of frames to be processed together for generating multi frame description
- sample_fps: frames per second to sample 
- resize_frames: boolean(True/False) decides whether you want to resize frames before passing them to VLM
- target_size: frame size after resizing frames
- model_name: model to be used for generating frame description (like smolVLM, gemini)
- output_dir: directory to be used for storing frame descriptions that are generated

```
#### Configuration Example ####

    base_model_id: HuggingFaceTB/SmolVLM2-2.2B-Instruct
    batch_size: 8
    max_frames: 600 
    preserve_json: True
    interval: 5
    use_vector_db: False
    show_stats: False
    frames_per_context: 5
    sample_fps: 1
    resize_frames: True
    target_size: 384
    model_name: smolVLM
    output_dir: ./results
    checkpoint_path:
```

### Config Properties for Audio Transcription(audio_cfg)

- asr_model: pipeline to be used for generating transcription (like whisper)
- diarization_model: model to be used for diarization (like pyannote/speaker-diarization-3.1)
- base_model_id: model hugging face id to be used for transcription generation (like openai/whisper-large-v3)
- language: language for transcription (like en for english)
- interval:  interval in sec to process together for generating audio trancription
- sampling_rate: sampling rate for extracting audio from the given video
- chunk_interval: chunk interval in sec to be used for chunking large videos for batch processing
- overlap_sec: interval in sec to be used for assigning to a give time frame 
- chunk_dir: temporary directory to be used for storing chunks of the video
- chunk_file_prefix: prefix for the filename to be used for storing a chunk
- output_dir:  directory to be used for storing audio trancription that are generated

```
#### Configuration Example ####
    asr_model: whisper
    diarization_model: pyannote/speaker-diarization-3.1
    base_model_id: openai/whisper-large-v3
    language: en
    interval: 5
    sampling_rate: 16000
    chunk_interval: 1200
    overlap_sec: 5
    chunk_dir: /tmp/
    chunk_file_prefix: chunk_
    output_dir: ./results
```


## Models currently supported
- Frame Description VLM
    - smolVLM
    - gemini
- Audio Transcription ASR
    - Pipeline
        - whisper
    - Diarization model
        - pyannote/speaker-diarization-*
    - ASR model
        - penai/whisper-*
- Query Engine LLM
    - OpenAI (like gpt-3.5-turbo, gpt-4 etc.)
    - Gemini (like gemini-2.0-flash etc.)

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- OpenCV
- YAML
- Other dependencies listed in essential-requirements.txt

## Advanced Configuration

For detailed statistics on VLM model usage, specify the `--show_stats` flag. This provides:

- Token usage statistics
- Processing time analysis
- Video metadata analysis
- Resource utilization metrics
- Estimated cost (if using commercial APIs)

## License

[MIT License](LICENSE)
