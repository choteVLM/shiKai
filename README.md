<p align="center">
  <h1 align="center"><b><i><u>Shikai</u></i></b>: A flexible search engine for videos</h1>
</p>
<p align="center">
  <img src="assets/title.png" alt="ShiKai Logo">
</p>


ShiKai is the most flexible video search engine out there! letting you pick and mix advanced vision and audio models for pinpoint search on your video database
Build your ideal video search stack by customizing every component from speech recognition and VLMs to retrieval workflows in an easy and scalable manner 🚀✨


## 📬 Contact Us

Need ShiKai tailored to your specific video processing needs? Our team can help with custom finetuning and integrations:

- **Shubham Sharma**: [shubham.sharma.c2023@iitbombay.org](mailto:shubham.sharma.c2023@iitbombay.org)
- **Aryan jain**: [aryan_j@cs.iitr.ac.in](mailto:aryan_j@cs.iitr.ac.in)


If you want to implement research papers in video models, contribute to the code development or help shape the [roadmap](#future-roadmap-), join our community!

<p align="center">
  <a href="https://discord.gg/v7ef76KXFc">
    <img src="https://img.shields.io/badge/Join%20our%20Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Join our Discord" width="250">
  </a>
</p>


## How ShiKai Works

ShiKai processes videos through three core pipelines that work together to enable natural language querying of video content:

### 🎬 Video Language Model (VLM) Pipeline

The VLM pipeline extracts visual information from your videos:
1. **Frame Extraction**: Videos are sampled at configurable intervals (e.g., 1 frame per second)
2. **Temporal Context**: Multiple frames are analyzed together to capture activities over time
3. **Dense captioninig**: Detailed text descriptions can generated for each frame or sequence depending on the model

You can choose between different VLMs like SmolVLM and Gemini Vision, or implement your own VLM interface.

### 🔊 Audio Speech Recognition (ASR) Pipeline

The ASR pipeline extracts spoken content from your videos:
1. **Audio Extraction**: Audio is separated from the video file
2. **Speaker Diarization**: Different speakers are identified and labeled
3. **Speech Recognition**: Audio is transcribed to text with timestamps
4. **Transcription Generation**: The final output includes speaker-identified text aligned with video timestamps

The default implementation uses Whisper for transcription and Pyannote for speaker diarization.

### 🔍 Query Engine

The Query Engine is the orchestration layer that determines how to analyze and understand video content with the help of the available models and tools:

1. **Model Selection and coordination**: Decides which models or tools are most appropriate for answering a specific query. Examples:
   - Using object detectors for "How many cars are visible in this traffic scene?"
   - Prioritizing ASR for "What did the presenter say about climate change?"
   - Combining face recognition with VLM for "When does this character appear again?"
2. **Temporal Alignment**: Aligns information across modalities with precise timestamps
3. **Knowledge Retrieval**: Can perform RAG (Retrieval Augmented Generation) over video embeddings to narrow down the search space on larger video databases 
4. **Adaptive Frame sampling**: Uses temporal search algorithms (like [T*](http://lvhaystackai.com)) to improve accuracy and speed of query responses 
5. **Response Generation**: Integrates the responses via a final chatLLM for the user

> **Note**: The current implementation uses a basic [Socratic model](https://socraticmodels.github.io) approach, where the video is parsed sequentially step by step and dense captions are generted. Future versions will support more sophisticated reasoning patterns, video RAG, parallel processing, and integration with specialized models like YOlO, SAM. 
##TODO: change this based on how much aryan is able to do 


## Getting started

### Repository Structure

```
/
├── main.py                      # Entry point for the application
├── VLM_main.py                  # Video Language Model (VLM) pipeline
├── ASR_main.py                  # Automatic Speech Recognition pipeline
├── configs/                     # Configuration files
│   ├── query_engine.yml         # Config for the main query engine
│   ├── vision_extract.yml       # Config for video models
│   └── audio_extract.yml        # Config for audio extraction  
├── chatLLM/                     # Chat models for query answering
│   ├── base_model.py            # Base interface for chat LLMs
│   ├── openAI.py                # OpenAI implementation
│   └── gemini.py                # Gemini Chat implementation
├── models_VLM/                # Video Language Models
│   ├── base_model.py            # Base interface for VLMs
│   ├── smolVLM.py               # SmolVLM implementation
│   └── gemini.py                # Gemini Vision implementation
├── models_ASR/                    # Audio Speech Recognition models
│   ├── base_model.py            # Base interface for ASR
│   └── whisper.py               # Whisper implementation
├── utils/                       # Utility functions
│   ├── video_utils.py           # Video processing utilities
│   ├── audio_utils.py           # Audio processing utilities
│   ├── db_utils.py              # Database utilities
│   ├── format_text_utils.py     # Text formatting utilities
│   └── index_utils.py           # Indexing utilities
├── viewer/                      # Web UI
│   └── video_chat.html          # HTML template for video chat interface
├── prompts/                     # Prompt templates
└── assets/                      # Images and other assets
```

### Installation


```bash
conda create -n shikai pip 
conda activate shikai
pip install -r shiKai/requirements.txt
```

### Setup API keys

ShiKai requires various API keys to function properly. You should set these as environment variables before running the application to avoid being prompted each time:

```bash
# For Hugging Face models (required for speaker diarization)
export HUGGING_FACE_TOKEN=your_hugging_face_token

# For Gemini API (if using Gemini for vision processing)
export GEMINI_VISION_API_KEY=your_gemini_vision_api_key

# For Gemini API (if using Gemini for chat)
export GEMINI_CHAT_API_KEY=your_gemini_chat_api_key

# For OpenAI API (if using OpenAI models)
export OPENAI_CHAT_API_KEY=your_openai_api_key
```
For persistent configuration, add these to your shell profile (e.g., `~/.bashrc`, `~/.zshrc`, etc.).

## Usage

```bash
python shiKai/main.py --video_cfg_file ./configs/query_engine.yml
```

### Web Interface

You can now use shiKai with a browser-based interface by adding the `--web` flag:

```bash
python shiKai/main.py --video_cfg_file ./configs/query_engine.yml --web
```

This will start a web server at http://localhost:5000 where you can:
- View the video
- Chat with the AI assistant about the video content
- Navigate through video clips

## Configuration

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
- web_mode: boolean(True/False) to run the application with a web interface instead of CLI

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
    web_mode: True
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


## Future Roadmap 🚀

### 🎯 Current Capabilities
- **Frame Description VLM**
    - smolVLM
    - Gemini Vision
- **Audio Transcription ASR**
    - Pipeline
        - Whisper
    - Diarization model
        - Pyannote/speaker-diarization-*
    - ASR model
        - OpenAI/whisper-*
- **Query Engine**
    - [Socratic model](https://socraticmodels.github.io) approach with OpenAI and Gemini models 

### 🔮 Coming Soon

#### 📊 Benchmarking & Evaluation
- **Long Video understanding Benchmarks**
  - [HourVideo](https://hourvideo.stanford.edu): Benchmarking differnt aspects like summarization, perception (recall, tracking), visual reasoning (spatial, temporal, predictive, causal, counterfactual), and navigation (room-to-room, object retrieval) in hour-long videos 
  - [MomentSeeker](https://huggingface.co/datasets/avery00/MomentSeeker): Benchmarking long-video moment retrieval (LVMR) with multimodal queries.
  - [LongVideoHaystack](https://huggingface.co/datasets/LVHaystack/LongVideoHaystack): A new benchmark of 3,874 human-annotated instances with fine-grained metrics for keyframe search quality and efficiency in videos up to 30 minutes
  - Support for standard evaluation metrics and comparison with SOTA methods

#### 🎬 Expanding Model Support
- **Advanced Vision Models**
  - Qwen2.5-VL: Alibaba's multimodal foundation model series 
  - OpenAI Vision models
  - Apollo models: multimodal models for video understanding released by meta
  - 12 Labs: Video understanding models like Marengo and Pegasus 
  
- **Specialized Models**
  - SAM: Segment Anything Model for precise object segmentation
  - YOLOWorld: Real-time object detection with open vocabulary
  - Face recognition models for person identification
  - Action recognition models for temporal activity tracking

#### 📚 RAG Pipeline for Video Databases
- **Video Embedding Models**
  - InternVideo2: State-of-the-art video representations (video encoder)
  - LangBind-Video: Language-bound video representations (video encoder)
  - SigLIP: Signal-to-language pre-training (image encoder)
  - V-JEPA: Video Joint Embedding Predictive Architecture trained with self-supervised learning (video encoder)
- **Database & Retrieval Improvements**
  - Vector database integration with Qdrant
  - Multi-modal embeddings for combined visual-audio search
  - Query optimization for large video collections
  - Semantic chunk-based retrieval for faster search

#### ⏱️ Advanced Search Algorithms
- **Adaptive Frame Sampling**
  - [T*](http://lvhaystackai.com):  Identify key frames relevant to specific queries by leveraging lightweight object detectors to reduce the inference budget of the VLM
  - [VideoTree](https://videotree2024.github.io): Build a tree structure in a query-adaptive manner for a coarse-to-fine search on frames
  - [VideoAgent](https://wxh1996.github.io/VideoAgent-Website/): Agent-based frame selection strategies using VLMs and other models as tools

- **Temporal Reasoning**
  - Causal reasoning across video segments
  - Event boundary detection
  - State tracking for objects and actors
  - Time-based query refinement


## License

This software is licensed under the Elastic License 2.0.

You may use, modify, and redistribute the code, but you may not provide the software as a managed service.  
See the [LICENSE](./LICENSE) file or visit [elastic.co/licensing](https://www.elastic.co/licensing/elastic-license) for details.
