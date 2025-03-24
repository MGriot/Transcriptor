import logging
import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import json
import os
from typing import List, Dict, Tuple, Optional
# https://huggingface.co/pyannote/speaker-diarization-3.1
# https://github.com/pyannote/pyannote-audio?tab=readme-ov-file
# Load the Hugging Face token from config.py
try:
    from config import HF_TOKEN
except ImportError:
    HF_TOKEN = None
    logging.warning(
        "config.py not found or HF_TOKEN not defined. Diarization may not work."
        " Please create a config.py file with HF_TOKEN = 'YOUR_HUGGINGFACE_TOKEN'"
    )

# Set up logging (optional, but good practice)
logging.basicConfig(level=logging.INFO)


def diarize_audio(
    audio_file_path: str,
    max_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
) -> Tuple[Pipeline, Dict]:
    """
    Performs speaker diarization on an audio file.

    Args:
        audio_file_path (str): Path to the audio file.
        max_speakers (int, optional): Maximum number of speakers. Defaults to None.
        min_speakers (int, optional): Minimum number of speakers. Defaults to None.

    Returns:
        tuple: A tuple containing the diarization pipeline and the diarization output.
    """
    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load the pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN,
    )
    pipeline.to(device)  # Move the pipeline to the selected device

    # Load the audio file
    try:
        waveform, sample_rate = torchaudio.load(audio_file_path)
    except Exception as e:
        logging.error(f"Error loading audio file: {e}")
        raise  # Re-raise the exception to be handled by the caller

    # Prepare the input for the pipeline
    input_data = {"waveform": waveform, "sample_rate": sample_rate}

    # Add speaker number hints if provided
    if max_speakers is not None:
        input_data["max_speakers"] = max_speakers
    if min_speakers is not None:
        input_data["min_speakers"] = min_speakers

    # Run the diarization pipeline with the progress hook
    with ProgressHook() as hook:
        try:
            diarization = pipeline(input_data, hook=hook)
        except Exception as e:
            logging.error(f"Error during diarization: {e}")
            raise  # Re-raise the exception

    logging.info("Diarization complete.")
    return pipeline, diarization  # Return both pipeline and diarization


def chunk_audio(
    audio_file_path: str, diarization: Dict, output_dir: str = "chunks"
) -> Dict:
    """
    Splits the audio file into chunks based on speaker diarization and saves them.

    Args:
        audio_file_path (str): Path to the audio file.
        diarization (Dict): The diarization output from the `diarize_audio` function.
        output_dir (str, optional): Directory to save the audio chunks. Defaults to "chunks".

    Returns:
        dict: A dictionary containing information about the saved chunks.  The keys are the speaker names,
              and the values are lists of dictionaries, where each dictionary contains the start time,
              end time, and the filename of the corresponding audio chunk.
              Example:
              {
                "speaker_1": [{"start": 0.5, "end": 1.5, "filename": "chunk_speaker_1_0.wav"}, ...],
                "speaker_2": [{"start": 2.0, "end": 3.0, "filename": "chunk_speaker_2_0.wav"}, ...]
              }
    """
    # Load the audio file
    try:
        waveform, sample_rate = torchaudio.load(audio_file_path)
    except Exception as e:
        logging.error(f"Error loading audio file: {e}")
        raise  # Re-raise the exception

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    chunk_data = {}
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start_sample = int(segment.start * sample_rate)
        end_sample = int(segment.end * sample_rate)
        chunk_waveform = waveform[:, start_sample:end_sample]

        # Create a unique filename for the chunk
        speaker_dir = os.path.join(output_dir, speaker)
        os.makedirs(speaker_dir, exist_ok=True)
        chunk_filename = f"chunk_{speaker}_{len(chunk_data.get(speaker, []))}.wav"
        chunk_filepath = os.path.join(speaker_dir, chunk_filename)

        # Save the chunk
        try:
            torchaudio.save(chunk_filepath, chunk_waveform, sample_rate)
            logging.info(f"Saved chunk: {chunk_filepath}")
        except Exception as e:
            logging.error(f"Error saving audio chunk: {e}")
            raise  # reraise

        # Store chunk information
        if speaker not in chunk_data:
            chunk_data[speaker] = []
        chunk_data[speaker].append(
            {"start": segment.start, "end": segment.end, "filename": chunk_filename}
        )

    return chunk_data


def process_audio_for_transcription(
    audio_file_path: str,
    max_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    output_dir: str = "chunks",
) -> str:
    """
    Performs speaker diarization and chunks the audio file, saving chunk information to a JSON file.

    Args:
        audio_file_path (str): Path to the audio file.
        max_speakers (int, optional): Maximum number of speakers. Defaults to None.
        min_speakers (int, optional): Minimum number of speakers. Defaults to None.
        output_dir (str, optional): Directory to save the audio chunks. Defaults to "chunks".
    Returns:
        str: The path to the JSON file containing the chunk information.
    """
    try:
        pipeline, diarization = diarize_audio(
            audio_file_path, max_speakers, min_speakers
        )
    except Exception as e:
        logging.error(f"Diarization failed: {e}")
        return None  # Explicitly return None in case of error

    try:
        chunk_data = chunk_audio(audio_file_path, diarization, output_dir)
    except Exception as e:
        logging.error(f"Chunking failed: {e}")
        return None  # Explicitly return None

    # Save chunk data to a JSON file
    json_filename = os.path.join(output_dir, "chunks.json")
    try:
        with open(json_filename, "w") as f:
            json.dump(chunk_data, f, indent=4)
        logging.info(f"Chunk data saved to: {json_filename}")
        return json_filename
    except Exception as e:
        logging.error(f"Error saving JSON file: {e}")
        return None  # error


if __name__ == "__main__":
    # Example usage: Set your audio file path here.
    audio_file_path = r"C:\Users\Admin\Documents\Coding\Transcriptor\audio\Botanicario - Ribes Nero.wav"  # Replace with your audio file
    max_speakers = 2  # Optional: Set the maximum number of speakers
    min_speakers = 1  # Optional: Set the minimum number of speakers
    output_dir = "audio_chunks"  # Directory where chunks will be saved

    json_file_path = process_audio_for_transcription(
        audio_file_path, max_speakers, min_speakers, output_dir
    )
    if json_file_path:
        print(f"Chunk information saved to: {json_file_path}")
    else:
        print("Failed to process audio for transcription.")
