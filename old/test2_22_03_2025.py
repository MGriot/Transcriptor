import logging
import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import json
import os
from typing import List, Dict, Tuple, Optional
import whisper
import datetime
from tqdm import tqdm  # Import tqdm for progress bar
import sys  # Import the sys module


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
        ):  # Add model name to desc
            # Add the 'words' key to each segment, even if it's an empty list.
            # This ensures consistency in the output structure.
            words = [
                {"start": word["start"], "end": word["end"], "word": word["word"]}
                for word in segment.get("words", [])
            ]
            segment_data = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "words": words,  # Include the words list here
            }
            logging.debug(f"Whisper output segment: {segment_data}")  # Log the segment
            transcript_data.append(segment_data)
        return transcript_data

    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        raise  # Re-raise the exception


def chunk_audio(audio_file_path: str, diarization: Dict, output_dir: str) -> Dict:
    """
    Splits the audio file into chunks based on speaker diarization and saves them.
    The chunks are saved into a single directory, and the function returns
    a dictionary with the chunk information, ordered by the start time of each chunk.

    Args:
        audio_file_path (str): Path to the audio file.
        diarization (Dict): The diarization output from the `diarize_audio` function.
        output_dir (str): Directory to save the audio chunks.

    Returns:
        dict: A dictionary containing information about the saved chunks, sorted by start time.
        The dictionary is a list of dictionaries, where each dictionary contains the
        start time, end time, speaker and the filename
        Example:
        [
            {"start": 0.5, "end": 1.5, "speaker":"speaker_1", "filename": "chunk_0.wav"},
            {"start": 2.0, "end": 3.0, "speaker":"speaker_2", "filename": "chunk_1.wav"},
            ...
        ]
    """
    # Load the audio file
    try:
        waveform, sample_rate = torchaudio.load(audio_file_path)
    except Exception as e:
        logging.error(f"Error loading audio file: {e}")
        raise  # Re-raise the exception

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    chunk_data: List[Dict] = []
    chunk_index = 0
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start_sample = int(segment.start * sample_rate)
        end_sample = int(segment.end * sample_rate)
        chunk_waveform = waveform[:, start_sample:end_sample]

        # Create a unique filename for the chunk
        chunk_filename = f"chunk_{chunk_index}.wav"
        chunk_filepath = os.path.join(output_dir, chunk_filename)

        # Save the chunk
        try:
            torchaudio.save(chunk_filepath, chunk_waveform, sample_rate)
            logging.info(f"Saved chunk: {chunk_filepath}")
        except Exception as e:
            logging.error(f"Error saving audio chunk: {e}")
            raise  # reraise

        # Store chunk information
        chunk_data.append(
            {
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker,
                "filename": chunk_filename,
            }
        )
        chunk_index += 1

    # Sort the chunks by start time
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
        output_dir = f"results/{file_name}_results"  # Changed default directory
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    combined_data = []  # Initialize combined_data
    logging.info(f"Processing audio file: {file_name}")  # Add file name
    json_files = []

    # 1.  If diarization is enabled, then perform diarization and chunking.
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
        chunk_transcriptions = {}  # Store transcriptions for each chunk
        for chunk in chunk_data:
            chunk_file_path = os.path.join(output_dir, chunk["filename"])
            try:
                chunk_transcript_data = transcribe_audio(
                    chunk_file_path, whisper_model_name
                )  # Transcribe each chunk
                chunk_transcriptions[chunk["filename"]] = (
                    chunk_transcript_data  # Store the transcription
                )
                # Create json file for each chunk
                chunk_json_filename = os.path.join(
                    output_dir, f"{chunk['filename'][:-4]}.json"
                )
                with open(chunk_json_filename, "w") as f:
                    json.dump(chunk_transcript_data, f, indent=4)
                json_files.append(chunk_json_filename)
                logging.info(f"Saved chunk json: {chunk_json_filename}")

            except Exception as e:
                logging.error(
                    f"Transcription failed for chunk: {chunk_file_path}. Error: {e}"
                )
                # Handle the error, e.g., add a placeholder.  Crucially, add speaker info.
                chunk_transcriptions[chunk["filename"]] = [
                    {  # Store a failed transcription
                        "start": chunk["start"],
                        "end": chunk["end"],
                        "text": "Transcription failed.",
                        "words": [],
                    }
                ]

        # Combine the transcription results with speaker information *after* all chunks are transcribed
        for chunk in chunk_data:
            chunk_transcription = chunk_transcriptions[chunk["filename"]]
            for segment in chunk_transcription:
                combined_data.append(
                    {
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"],
                        "speaker": chunk["speaker"],  # from the chunk data
                        "words": segment["words"],
                    }
                )
        combined_data.sort(key=lambda x: x["start"])  # Sort the *combined* data

    else:
        # 2. Transcribe the audio (the whole file)
        try:
            transcript_data = transcribe_audio(audio_file_path, whisper_model_name)
            combined_data = transcript_data
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
    current_end_time = None

    for entry in data:
        speaker = entry["speaker"]
        start_time = entry["start"]
        end_time = entry["end"]
        text = entry["text"]

        if speaker == current_speaker:
            current_text += " " + text
            current_end_time = end_time
        else:
            if current_speaker is not None:
                transcript += f"{current_speaker} ({current_start_time:.2f} - {current_end_time:.2f}): {current_text}\n"
            current_speaker = speaker
            current_start_time = start_time
            current_text = text
            current_end_time = end_time

    # Add the last speaker's text
    if current_speaker is not None:
        transcript += f"{current_speaker} ({current_start_time:.2f} - {current_end_time:.2f}): {current_text}\n"
    return transcript


if __name__ == "__main__":
    audio_file_path = r"C:\Users\Admin\Documents\Coding\Transcriptor\audio\test.mp3"
    enable_diarization = True  # Or False, depending on whether you want diarization
    max_speakers = 2
    min_speakers = 1
    output_dir = "out"  # changed output directory
    whisper_model_name = "medium"

    json_file_path = process_audio_for_transcription(  # Corrected function name here
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
        print(transcript)  # Print to the console
        # Optionally save to a text file
        txt_file_path = os.path.join(output_dir, "transcript.txt")
        try:
            with open(
                txt_file_path, "w", encoding="utf-8"
            ) as f:  # Specify UTF-8 encoding
                f.write(transcript)
            print(f"Transcript saved to: {txt_file_path}")
        except Exception as e:
            logging.error(f"Error saving transcript to file: {e}")
    else:
        print("Failed to process audio.")
