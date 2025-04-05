# SmolVLM Video Inference

A Python package for extracting frames from videos and running SmolVLM visual language model inference on them. This tool generates detailed descriptions of video frames and creates a summarized world state history.

## Features

- Extract frames from videos at configurable intervals
- Process frames with SmolVLM2 visual language models (256M, 2.2B, etc.)
- Generate detailed frame descriptions with customizable prompts
- Create structured world state history with temporal organization
- Optional vector database integration for frame embeddings (Qdrant)
- Multi-frame context processing for sequential frame analysis
- Detailed statistics tracking for token usage and performance

## Installation

### Option 1: Install from source (Development Mode)

```bash
git clone https://github.com/yourname/smolVLM.git
cd smolVLM
pip install -e .
```

### Option 2: Install Core Dependencies Only

If you want to skip installing the package's dependencies and only install core requirements:

```bash
pip install -e . --no-deps
pip install -r smolVLM/essential-requirements.txt
```

## Usage

```bash
python smolVLM/SmolVLM_video_inference.py --video_path /path/to/your/video.mp4
```

### Basic Options

- `--video_path`: Path to the video file to process
- `--prompt_path`: Path to YAML file containing the prompt (default: caption.yaml)
- `--base_model`: Base model ID to use (default: HuggingFaceTB/SmolVLM2-2.2B-Instruct)
- `--batch_size`: Number of frames to process in each batch (default: 7)
- `--max_frames`: Maximum number of frames to extract (default: 60)
- `--interval_seconds`: Time interval for grouping frames in history (default: 5)
- `--use_vector_db`: Enable vector database for frame embeddings
- `--show_stats`: Display detailed statistics about processing

### Multi-Frame Context Processing

The package now supports processing multiple frames in a single context, allowing the model to analyze frame sequences rather than individual frames:

```bash
python smolVLM/SmolVLM_video_inference.py --video_path /path/to/video.mp4 --multi_frame_context --frames_per_context 3
```

#### Multi-Frame Options

- `--multi_frame_context`: Process multiple frames in a single context
- `--frames_per_context`: Number of frames to include in each context (default: 3)

This mode is ideal for analyzing motion, temporal relationships, and sequential changes in videos. The model receives multiple frames at once and can provide analysis of their relationships, changes, and continuity.

## Prompt Templates

Two main prompt templates are available:

1. **caption.yaml**: Default template for single-frame processing
2. **multi_frame.yaml**: Template optimized for multi-frame context processing

You can create custom prompts by following these templates. Available placeholder variables:

- `{{FRAMES_COUNT}}`: Total number of frames being processed
- `{{TIME_INTERVAL}}` and `{{TIME_INTERVAL_FORMATTED}}`: Time between frames
- `{{TOTAL_DURATION}}` and `{{TOTAL_DURATION_FORMATTED}}`: Total video duration
- `{{FRAMES_PER_CONTEXT}}`: Number of frames in each context (multi-frame mode)
- `{{CONTEXT_INTERVAL}}` and `{{CONTEXT_INTERVAL_FORMATTED}}`: Time span covered by each context

## Examples

### Basic Single-Frame Processing

```bash
python smolVLM/SmolVLM_video_inference.py --video_path /path/to/video.mp4 --max_frames 60 --show_stats
```

### Multi-Frame Sequence Analysis

```bash
python smolVLM/SmolVLM_video_inference.py --video_path /path/to/video.mp4 --multi_frame_context --frames_per_context 3 --show_stats
```

### Processing with Vector Database for Embeddings

```bash
python smolVLM/SmolVLM_video_inference.py --video_path /path/to/video.mp4 --use_vector_db
```

## Output

The script generates two files in the `results` directory:

1. `<video_name>_world_state_history.json`: JSON representation of video world state
2. `<video_name>_world_state_text.txt`: Human-readable text representation

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- OpenCV
- YAML
- Other dependencies listed in setup.py

## Advanced Configuration

For detailed statistics on model usage, specify the `--show_stats` flag. This provides:

- Token usage statistics
- Processing time analysis
- Video metadata analysis
- Resource utilization metrics
- Estimated cost (if using commercial APIs)

## License

[MIT License](LICENSE)