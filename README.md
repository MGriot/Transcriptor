# Audio Transcriptor

A comprehensive Python-based audio transcription and speaker diarization tool supporting a wide range of audio/video formats. Features both CLI and GUI interfaces for flexible usage.

## Features

### Input Format Support
- **Audio Formats**: `.wav`, `.mp3`, `.ogg`, `.flac`, `.aac`, `.m4a`, `.wma`, `.aif`, `.aiff`
- **Video Formats**: `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.flv`
- Automatic format conversion with quality preservation
- Batch processing capabilities

### Core Functionality
- **Transcription**: OpenAI's Whisper engine with multiple model options
- **Speaker Diarization**: pyannote.audio-based speaker separation
- **Voice Activity Detection (VAD)**: Multiple methods including Silero and Auditok
- **Language Support**: Auto-detection and 99+ language codes
- **Interactive Speaker Naming**: Easy post-processing speaker identification
- **Word-level Timestamps**: Precise timing for each transcribed word

### Advanced Features
- GPU acceleration support (CUDA)
- Parallel processing for batch operations
- Progress tracking and logging
- Format conversion with quality preservation
- Error recovery and session persistence

## System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (recommended 4+ cores)
- **RAM**: Minimum 8GB (16GB recommended)
- **GPU**: NVIDIA GPU with CUDA support (optional, but recommended)
- **Storage**: 2GB minimum for installation, plus space for audio files

### Software Requirements
- Python 3.8+ (3.10 recommended)
- NVIDIA CUDA Toolkit 11.0+ (for GPU acceleration)
- FFmpeg (required for audio/video processing)

## Installation

1. **System Prerequisites**:
```bash
# Windows (using Chocolatey)
choco install ffmpeg python3

# Linux (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install ffmpeg python3 python3-venv
```

2. **Project Setup**:
```bash
# Clone repository
git clone https://github.com/yourusername/Transcriptor.git
cd Transcriptor

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

3. **Hugging Face Setup**:
- Create account at [Hugging Face](https://huggingface.co)
- Accept pyannote/speaker-diarization-3.1 [model terms](https://huggingface.co/pyannote/speaker-diarization-3.1)
- Generate token at [Settings/Tokens](https://huggingface.co/settings/tokens)
- Create `config.json`:
```json
{
    "hf_token": "your_token_here"
}
```

## Usage

### GUI Interface

Launch with:
```bash
python gui.py
```

Features:
1. **File Management**:
   - Single/Multiple file selection
   - Drag-and-drop support
   - Output directory management

2. **Transcription Settings**:
   - Model selection (tiny to large)
   - Language configuration
   - Speaker count limits
   - VAD method selection

3. **Processing Controls**:
   - Start/Stop functionality
   - Progress monitoring
   - Real-time logging

4. **Speaker Management**:
   - Interactive speaker naming
   - Batch speaker updates
   - Name template support

### CLI Interface

1. **Basic Usage**:
```bash
python cli.py path/to/audio_file
```

2. **Advanced Usage**:
```bash
python cli.py path/to/audio_file \
  --output-dir "output" \
  --whisper-model "base" \
  --language "en" \
  --min-speakers 2 \
  --max-speakers 4 \
  --use-vad \
  --vad-method "silero" \
  --verbose
```

3. **Batch Processing**:
```bash
python cli.py path/to/directory \
  --output-dir "batch_output" \
  --num-processes 4
```

### Python API Usage

```python
from audio_transcriber import AudioTranscriber

# Basic usage
transcriber = AudioTranscriber(
    audio_file="input.mp3",
    output_dir="output",
    hf_token="your_token"
)

# Process with advanced options
transcriber.process_audio(
    language="en",
    min_speakers=2,
    max_speakers=4,
    use_vad=True,
    vad_method="silero",
    num_processes=4
)
```

## Output Structure

The tool generates several output files in your specified output directory:

- `{audio_name}_transcriptions.json`: Raw transcription data
- `{audio_name}_diarized_transcription.txt`: Clean transcription with speaker labels
- `{audio_name}_speaker_names.json`: Speaker naming information
- `chunks/`: Directory containing audio segments (when using diarization)

## Configuration

You can create a `config.json` file to store your Hugging Face token and other default settings:

```json
{
    "hf_token": "your_token_here"
}
```

## Requirements

- Python 3.8+ (3.10 recommended)
- PyTorch
- torchaudio
- pyannote.audio
- whisper_timestamped
- pydub
- tkinter (for GUI)
- FFmpeg (required for audio/video processing)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [whisper_timestamped](https://github.com/linto-ai/whisper-timestamped)
