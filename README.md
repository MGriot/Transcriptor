# Audio Transcriptor

A powerful audio transcription tool with speaker diarization capabilities, built using Python. This tool can transcribe audio files, identify different speakers, and provide detailed timestamps for each utterance.

## Features

- **Audio Transcription**: Uses OpenAI's Whisper models for accurate speech-to-text conversion
- **Speaker Diarization**: Identifies and separates different speakers in the audio
- **Multiple Interface Options**: 
  - Command Line Interface (CLI)
  - Graphical User Interface (GUI)
- **Voice Activity Detection (VAD)**: Multiple VAD methods supported:
  - Silero
  - Silero 3.1
  - Auditok
- **Flexible Input Handling**:
  - Single audio file processing
  - Batch processing of multiple files
  - Supports multiple audio formats (.wav, .mp3, .ogg, .flac, .aac)
- **Customizable Output**:
  - JSON output with detailed information
  - Clean text output with timestamps
  - Speaker labels and custom naming
- **Advanced Features**:
  - Multi-language support
  - Word alignment plotting
  - Disfluency detection
  - Parallel processing capabilities

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Transcriptor.git
cd Transcriptor
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up Hugging Face API token:
- Create an account at [Hugging Face](https://huggingface.co)
- Get your API token from https://huggingface.co/settings/tokens
- Either:
  - Create a `config.json` file with your token:
    ```json
    {
        "hf_token": "your_token_here"
    }
    ```
  - Or provide it when running the program

## Usage

### Using the GUI

Run the graphical interface:
```bash
python audio_transcriber_gui.py
```

The GUI provides easy access to all features through a user-friendly interface.

### Using the CLI

Basic usage:
```bash
python cli.py path/to/audio_file
```

Advanced usage with options:
```bash
python cli.py path/to/audio_file \
  --output_dir output \
  --whisper_model base \
  --language en \
  --use_vad \
  --vad_method silero \
  --verbose
```

### Command Line Arguments

- `audio_input`: Path to audio file or directory
- `--output_dir`: Output directory (default: "output")
- `--hf_token`: Hugging Face API token
- `--skip_diarization`: Skip speaker diarization
- `--whisper_model`: Choose model size (tiny/base/small/medium/large)
- `--language`: Specify language code (e.g., en, fr, es)
- `--min_speakers`: Minimum number of speakers
- `--max_speakers`: Maximum number of speakers
- `--use_vad`: Enable Voice Activity Detection
- `--vad_method`: Choose VAD method
- `--verbose`: Enable verbose output
- `--plot_word_alignment`: Enable word alignment plotting
- `--detect_disfluencies`: Enable disfluency detection
- `--no_rename_speakers`: Disable interactive speaker renaming
- `--num_processes`: Number of processes for parallel processing

## Output Files

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

- Python 3.8+
- PyTorch
- torchaudio
- pyannote.audio
- whisper_timestamped
- pydub
- tkinter (for GUI)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [whisper_timestamped](https://github.com/linto-ai/whisper-timestamped)
