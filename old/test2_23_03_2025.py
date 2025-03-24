import logging
import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import json
import os
from typing import List, Dict, Tuple, Optional
import whisper
from tqdm import tqdm

# Load the Hugging Face token from config.py
try:
    from config import HF_TOKEN
except ImportError:
    HF_TOKEN = None
    logging.warning(
        "config.py not found or HF_TOKEN not defined. Diarization may not work."
        " Please create a config.py file with HF_TOKEN = 'YOUR_HUGGINGFACE_TOKEN'"
    )

# Set up logging
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


def transcribe_audio(audio_file_path: str, model_name: str = "base") -> List[Dict]:
    """
    Transcribes the audio file using Whisper.

    Args:
        audio_file_path (str): Path to the audio file.
        model_name (str): The name of the Whisper model to use (e.g., "base", "small", "medium", "large").
            Defaults to "base".

    Returns:
        List[Dict]: A list of transcription segments, where each segment is a dictionary
                        containing start time, end time, and transcribed text.
    """
    # Load the Whisper model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model(model_name, device=device)
    logging.info(f"Using Whisper model: {model_name} on device: {device}")

    # Transcribe the audio file
    try:
        result = model.transcribe(audio_file_path, word_timestamps=True)
        segments = result["segments"]
        logging.info("Transcription complete.")

        # Convert the segments to the desired format
        transcript_data = []
        for segment in tqdm(
            segments,
            desc=f"Transcribing {os.path.basename(audio_file_path)} with {model_name}",
        ):
            words = [
                {"start": word["start"], "end": word["end"], "word": word["word"]}
                for word in segment.get("words", [])
            ]
            segment_data = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "words": words,
            }
            logging.debug(f"Whisper output segment: {segment_data}")
            transcript_data.append(segment_data)
        return transcript_data

    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        raise  # Re-raise the exception


def chunk_audio(audio_file_path: str, diarization: Dict, output_dir: str) -> Dict:
    """
    Splits the audio file into chunks based on speaker diarization and saves them.
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_file_path)
    except Exception as e:
        logging.error(f"Error loading audio file: {e}")
        raise

    os.makedirs(output_dir, exist_ok=True)

    chunk_data: List[Dict] = []
    chunk_index = 0

    # First, collect all segments with their speaker information
    segments_with_speakers = [
        (segment, speaker)
        for segment, _, speaker in diarization.itertracks(yield_label=True)
    ]

    # Process each segment
    for segment, speaker in segments_with_speakers:
        start_sample = int(segment.start * sample_rate)
        end_sample = int(segment.end * sample_rate)
        chunk_waveform = waveform[:, start_sample:end_sample]

        # Create filenames for audio and JSON
        audio_filename = f"chunk_{chunk_index}.wav"
        audio_filepath = os.path.join(output_dir, audio_filename)

        # Save the audio chunk
        try:
            torchaudio.save(audio_filepath, chunk_waveform, sample_rate)
            logging.info(f"Saved audio chunk: {audio_filepath}")

            # Create initial JSON data for this chunk
            chunk_info = {
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker,
                "filename": audio_filename,
                "text": "",  # Will be filled by transcription
                "words": [],  # Will be filled with word timestamps
            }

            # Add to our complete chunk data
            chunk_data.append(chunk_info)

        except Exception as e:
            logging.error(f"Error saving chunk {chunk_index}: {e}")
            raise

        chunk_index += 1

    # Sort chunks by start time
    chunk_data.sort(key=lambda x: x["start"])

    return chunk_data


def process_audio_for_transcription(
    audio_file_path: str,
    enable_diarization: bool = False,
    max_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    output_dir: Optional[str] = None,
    whisper_model_name: str = "base",
) -> Optional[str]:
    """
    Performs speaker diarization (optional) and transcription on an audio file,
    and saves the results to a JSON file.

    Args:
        audio_file_path (str): Path to the audio file.
        enable_diarization (bool, optional): Whether to perform speaker diarization. Defaults to False.
        max_speakers (int, optional): Maximum number of speakers. Defaults to None.
        min_speakers (int, optional): Minimum number of speakers. Defaults to None.
        output_dir (str, optional): Directory to save the audio chunks and JSON file.
            If None, a default directory will be created.
        whisper_model_name (str, optional): The name of the Whisper model to use. Defaults to "base".

    Returns:
        Optional[str]: The path to the JSON file containing the combined results, or None on error.
    """
    # Extract the filename from the audio path
    file_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    # Use the filename to create the output directory
    if output_dir is None:
        output_dir = f"results/{file_name}_results"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    combined_data = []  # Initialize combined_data
    logging.info(f"Processing audio file: {file_name}")

    # 1. If diarization is enabled, then perform diarization and chunking.
    if enable_diarization:
        try:
            pipeline, diarization = diarize_audio(
                audio_file_path, max_speakers, min_speakers
            )
        except Exception as e:
            logging.error(f"Diarization failed: {e}")
            return None

        chunk_data = chunk_audio(
            audio_file_path, diarization, output_dir
        )  # chunk audio

        # 2. Transcribe the audio (in chunks if diarization is enabled)
        for chunk in chunk_data:
            chunk_file_path = os.path.join(output_dir, chunk["filename"])
            try:
                chunk_transcript_data = transcribe_audio(
                    chunk_file_path, whisper_model_name
                )
                chunk["text"] = (
                    chunk_transcript_data[0]["text"]
                    if chunk_transcript_data
                    else "Transcription failed."
                )
                chunk["words"] = (
                    chunk_transcript_data[0]["words"] if chunk_transcript_data else []
                )
            except Exception as e:
                logging.error(
                    f"Transcription failed for chunk: {chunk_file_path}. Error: {e}"
                )
                chunk["text"] = "Transcription failed."
                chunk["words"] = []

            combined_data.append(chunk)

    else:
        # 2. Transcribe the audio (the whole file)
        try:
            transcript_data = transcribe_audio(audio_file_path, whisper_model_name)
            combined_data.extend(transcript_data)
        except Exception as e:
            logging.error(f"Transcription failed: {e}")
            return None

    # 3. Save the combined data to a JSON file
    json_filename = os.path.join(output_dir, "combined_results.json")
    try:
        with open(json_filename, "w") as f:
            json.dump(combined_data, f, indent=4)
        logging.info(f"Combined data saved to: {json_filename}")
        return json_filename
    except Exception as e:
        logging.error(f"Error saving JSON file: {e}")
        return None


def generate_transcript_from_json(json_file_path: str) -> str:
    """
    Generates a formatted transcript from a JSON file, merging text from the same speaker
    at the segment level.  Word-level information is ignored for merging.

    Args:
        json_file_path (str): Path to the JSON file containing the transcription data.

    Returns:
        str: A formatted transcript string.
    """
    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON file: {e}")
        return "Error: Could not load JSON file."

    transcript = ""
    current_speaker = None
    current_start_time = None
    current_text = ""

    for entry in data:
        speaker = entry["speaker"]
        start_time = entry["start"]
        text = entry["text"]

        if speaker == current_speaker:
            current_text += " " + text
        else:
            if current_speaker is not None:
                transcript += (
                    f"{current_speaker} ({current_start_time:.2f}s): {current_text}\n"
                )
            current_speaker = speaker
            current_start_time = start_time
            current_text = text

    # Add the last speaker's text
    if current_speaker is not None:
        transcript += f"{current_speaker} ({current_start_time:.2f}s): {current_text}\n"
    return transcript


if __name__ == "__main__":
    audio_file_path = r"C:\Users\Admin\Documents\Coding\Transcriptor\audio\test.mp3"
    enable_diarization = True  # Set to True if you want to enable diarization
    max_speakers = 2
    min_speakers = 1
    output_dir = "out"
    whisper_model_name = "medium"

    json_file_path = process_audio_for_transcription(
        audio_file_path,
        enable_diarization,
        max_speakers,
        min_speakers,
        output_dir,
        whisper_model_name,
    )
    if json_file_path:
        print(f"Results saved to: {json_file_path}")
        transcript = generate_transcript_from_json(json_file_path)
        print("\nTranscript:\n")
        print(transcript)

        # Optionally save to a text file
        txt_file_path = os.path.join(output_dir, "transcript.txt")
        try:
            with open(txt_file_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            print(f"Transcript saved to: {txt_file_path}")
        except Exception as e:
            logging.error(f"Error saving transcript to file: {e}")
    else:
        print("Failed to process audio.")
