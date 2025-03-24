# Description: Test file for the 19th of March 2025
# voglio creare una funzione o classe in python per fare la trascrizione di file audio mediante la libreria Whisper: # https://github.com/openai/whisper
# Vorrei che lo script accettasse in input un file audio e restituisse la trascrizione del file audio.
# Vorrei che riconoscesse in automatico la linga parlata ma che possa anche essere passata come variabile.
# https://gemini.google.com/app/bd8806cd36c22854?hl=en
# i want to add speaker diarization for split the track audio into chenks based on timestamps and speaker, after i wnat the trasctiption of each chunck

import logging
import os
import whisper
import torch
from pyannote.audio import Pipeline
import soundfile as sf
import numpy as np
import shutil  # Import the shutil module
from tqdm import tqdm  # Import tqdm for progress bars


# Set up logging for any warnings or errors
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load the Hugging Face token from config.py
try:
    from config import HF_TOKEN
except ImportError:
    HF_TOKEN = None
    logging.warning(
        "config.py not found or HF_TOKEN not defined. Diarization may not work."
        " Please create a config.py file with HF_TOKEN = 'YOUR_HUGGINGFACE_TOKEN'"
    )


class AudioTranscriber:
    """
    A class to transcribe audio files using OpenAI's Whisper model, with speaker diarization.
    """

    def __init__(
        self,
        whisper_model_name="base",
        diarization_pipeline="pyannote/speaker-diarization",  # Updated version
    ):
        """
        Initializes the AudioTranscriber object with a pre-loaded Whisper model and diarization pipeline.

        Args:
            whisper_model_name (str, optional): The name of the Whisper model to use
                (e.g., "tiny", "base", "small", "medium", "large").
                Defaults to "base". Larger models are more accurate but require more resources.
            diarization_pipeline (str, optional):  The name or path of the pyannote.audio pipeline.
                Defaults to "pyannote/speaker-diarization".  # Updated version
        """
        try:
            # Try to load the specified Whisper model.
            self.whisper_model = whisper.load_model(whisper_model_name)
            self.whisper_model_name = whisper_model_name  # save the model name
            logging.info(f"Whisper model '{whisper_model_name}' loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load Whisper model '{whisper_model_name}': {e}")
            raise

        try:
            # Initialize the Pyannote diarization pipeline.  This requires an access token.
            # The token is read automatically from ~/.config/pyannote/access_token.
            if HF_TOKEN:
                try:
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        diarization_pipeline, use_auth_token=HF_TOKEN
                    )
                    self.diarization_pipeline.to(
                        torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    )
                    logging.info(
                        f"Diarization pipeline '{diarization_pipeline}' loaded successfully."
                    )
                except OSError as e:
                    if e.winerror == 1314:
                        logging.warning(
                            "Failed to create a symbolic link (error 1314).  "
                            "This is a common issue on Windows.  Attempting to copy the file instead."
                        )
                        # Attempt to use a copy strategy instead of a symlink.
                        import speechbrain.utils.fetching as fetching

                        original_fetch = fetching.fetch

                        def fetch_with_copy(
                            url,
                            dest,
                            filename=None,
                            save_dir=None,
                            force_download=False,
                            progressbar=True,
                            verifier=None,
                            local_strategy="copy",  # Change the default strategy to 'copy'
                        ):
                            return original_fetch(
                                url,
                                dest,
                                filename=filename,
                                save_dir=save_dir,
                                force_download=force_download,
                                progressbar=progressbar,
                                verifier=verifier,
                                local_strategy=local_strategy,
                            )

                        fetching.fetch = fetch_with_copy

                        self.diarization_pipeline = Pipeline.from_pretrained(
                            diarization_pipeline, use_auth_token=HF_TOKEN
                        )
                        self.diarization_pipeline.to(
                            torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        )
                        logging.info(
                            f"Diarization pipeline '{diarization_pipeline}' loaded successfully using copy strategy."
                        )
                    else:
                        raise e  # Re-raise the exception if it's not a 1314 error.

            else:
                self.diarization_pipeline = None
                logging.warning(
                    "Hugging Face token not found. Speaker diarization will be disabled."
                )
        except Exception as e:
            logging.error(
                f"Failed to load diarization pipeline '{diarization_pipeline}': {e}"
            )
            raise

        self.speaker_names = {}  # Dictionary to store speaker names

    def diarize_audio(self, audio_file_path):
        """
        Performs speaker diarization on the provided audio file.

        Args:
            audio_file_path (str): The path to the audio file to diarize.

        Returns:
            pyannote.core.Annotation: The diarization result.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            Exception: If an error occurs during diarization.
        """
        if not os.path.exists(audio_file_path):
            logging.error(f"Audio file not found: {audio_file_path}")
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        if self.diarization_pipeline is None:
            logging.warning(
                "Diarization pipeline is not available. Skipping diarization."
            )
            return None

        try:
            # Perform diarization.
            logging.info(f"Performing diarization on '{audio_file_path}'...")
            # Create a progress bar for diarization
            with tqdm(
                total=100, desc="Diarization"
            ) as pbar:  # Total progress is arbitrary 100.
                diarization = self.diarization_pipeline(audio_file_path)
                # Simulate progress.  In a real scenario, the pipeline would
                # ideally provide progress information.
                for _ in range(100):
                    pbar.update(1)  # Update progress bar
            logging.info("Diarization completed.")
            return diarization
        except Exception as e:
            logging.error(f"Error during diarization of '{audio_file_path}': {e}")
            raise Exception(f"Error during audio file diarization: {e}") from e

    def transcribe_audio(self, audio_file_path, language=None, diarization=None):
        """
        Transcribes the provided audio file, optionally with speaker diarization.

        Args:
            audio_file_path (str): The path to the audio file to transcribe.
            language (str, optional): The language to use for transcription (e.g., "it", "en", "fr").
                If None, the language is automatically detected. Defaults to None.
            diarization (pyannote.core.Annotation, optional): A pre-computed diarization.
                If None, it will be computed internally.

        Returns:
            list: A list of dictionaries, where each dictionary contains the speaker,
                start time, end time, and transcribed text.
                Returns an empty list if the transcription fails.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            Exception: If an error occurs during transcription.
        """
        if not os.path.exists(audio_file_path):
            logging.error(f"Audio file not found: {audio_file_path}")
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        try:
            if diarization is None:
                # Perform diarization if not provided.
                if self.diarization_pipeline is not None:
                    diarization = self.diarize_audio(audio_file_path)
                else:
                    diarization = None  # Set to None if pipeline was not loaded.

            # Load the audio file.
            audio, sample_rate = sf.read(audio_file_path)

            # Perform the transcription with the specified language (if provided).
            if language:
                logging.info(
                    f"Transcribing '{audio_file_path}' with forced language: {language}"
                )
            else:
                logging.info(
                    f"Transcribing '{audio_file_path}' with automatic language detection."
                )

            # Check if the device is CPU and force FP32 if necessary
            if self.whisper_model.device == torch.device("cpu"):
                transcription = self.whisper_model.transcribe(
                    audio_file_path, language=language, fp16=False
                )  # Force FP32
            else:
                transcription = self.whisper_model.transcribe(
                    audio_file_path, language=language
                )
            segments = transcription["segments"]

            # Create a progress bar for transcription
            with tqdm(total=len(segments), desc="Transcription") as pbar:
                results = []
                # Check if segments is None or not iterable
                if segments is None:
                    logging.error(
                        "Transcription segments are None.  This is likely due to an error in the transcription process."
                    )
                    return []  # Return an empty list to avoid the 'NoneType' error

                if not isinstance(segments, list):
                    logging.error(
                        "Transcription segments is not a list.  Expected a list of segments."
                    )
                    return []

                if not segments:  # Check if segments is empty
                    logging.warning(
                        "Transcription segments is empty.  No transcription to process."
                    )
                    return []  # Return an empty list for empty transcription

                for segment in segments:
                    start_time = segment["start"]
                    end_time = segment["end"]
                    text = segment["text"]

                    # Find the speaker(s) active during the segment.
                    speakers = []
                    if diarization:  # Only if diarization was performed
                        for turn in diarization.crop(
                            segment["start"], segment["end"]
                        ):  # Use segment as a time range
                            speakers.append(turn.label)
                    if speakers:
                        speaker = ", ".join(
                            speakers
                        )  # Join multiple speakers if needed
                    else:
                        speaker = "Unknown"

                    results.append(
                        {
                            "speaker": speaker,
                            "start_time": start_time,
                            "end_time": end_time,
                            "text": text,
                        }
                    )
                    pbar.update(1)  # Update progress bar

            logging.info("Transcription completed.")
            return results

        except Exception as e:
            logging.error(f"Error during transcription of '{audio_file_path}': {e}")
            return []  # Return empty list

    def get_transcription_with_speaker_names(
        self, transcription_results, speaker_names=None
    ):
        """
        Returns the transcription results with speaker names.

        Args:
            transcription_results (list): A list of transcription dictionaries
                (output of transcribe_audio).
            speaker_names (dict, optional): A dictionary mapping original speaker IDs
                (e.g., "SPEAKER_01") to new speaker names (e.g., "Alice").
                If None, the original speaker IDs will be used. Defaults to None.

        Returns:
            list: A new list of transcription dictionaries with speaker names.
        """
        updated_results = []
        if transcription_results is None:
            logging.warning(
                "Transcription results is None in get_transcription_with_speaker_names."
            )
            return []
        for segment in transcription_results:
            original_speaker = segment["speaker"]
            # Handle cases where multiple speakers are in the same segment
            speakers = [
                s.strip() for s in original_speaker.split(",")
            ]  # get a list of speakers
            new_speakers = []
            for sp in speakers:
                if speaker_names and sp in speaker_names:
                    new_speakers.append(speaker_names[sp])
                else:
                    new_speakers.append(
                        sp
                    )  # keep original speaker name if not found in speaker_names or if speaker_names is None
            new_speaker_name = ", ".join(new_speakers)  # join the speakers

            updated_segment = {
                "speaker": new_speaker_name,  # Use the replaced name
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "text": segment["text"],
            }
            updated_results.append(updated_segment)
        return updated_results

    def process_audio(self, audio_file_path, language=None, speaker_names=None):
        """
        Transcribes the audio file and applies speaker name replacement.

        Args:
            audio_file_path (str): Path to the audio file.
            language (str, optional): Language code (e.g., 'en', 'es').  Defaults to None.
            speaker_names (dict, optional): Dictionary of speaker names.
                e.g., {"SPEAKER_01": "Alice", "SPEAKER_02": "Bob"}. Defaults to None.

        Returns:
            list: A list of transcription dictionaries with speaker names.
        """
        transcription = self.transcribe_audio(audio_file_path, language)
        self.transcription_result = transcription  # Store the initial result
        return self.get_transcription_with_speaker_names(transcription, speaker_names)

    def update_speaker_names(self, speaker_names):
        """
        Updates the speaker names in the stored transcription result.

        Args:
            speaker_names (dict): A dictionary mapping original speaker IDs
                (e.g., "SPEAKER_01") to new speaker names (e.g., "Alice").
        """
        if hasattr(
            self, "transcription_result"
        ):  # Check if transcription has been done
            self.transcription_result = self.get_transcription_with_speaker_names(
                self.transcription_result, speaker_names
            )
        else:
            logging.warning("Transcription has not been performed yet.")

    def get_updated_transcription(self):
        """
        Returns the transcription result with the current speaker names.

        Returns:
            list: A list of transcription dictionaries with speaker names.
        """
        if hasattr(self, "transcription_result"):
            return self.transcription_result
        else:
            logging.warning("Transcription has not been performed yet.")
            return None


def main():
    """
    Main function to execute the transcription of a sample audio file.
    """
    # Example usage of the AudioTranscriber class:
    sample_audio_file = r"C:\Users\Admin\Documents\Coding\Transcriptor\audio\4547.mp3"  # Replace with your audio file path
    # Create an empty audio file if it does not exist
    if not os.path.exists(sample_audio_file):
        with open(sample_audio_file, "w") as f:
            pass

    # Create an instance of the AudioTranscriber class with the "base" model.
    # You need to have a valid access token for pyannote.audio to use diarization.
    transcriber = AudioTranscriber(
        whisper_model_name="base"
    )  # You can change the model here

    try:
        # 1.  Process audio with default speaker names.
        default_transcription = transcriber.process_audio(sample_audio_file)
        print("\nTranscription with Default Speaker Names:")
        print(default_transcription)

        # 2.  Update speaker names and get the updated transcription.
        speaker_name_mapping = {
            "SPEAKER_00": "Alice",
            "SPEAKER_01": "Bob",
            "SPEAKER_02": "Charlie",
        }
        transcriber.update_speaker_names(speaker_name_mapping)
        updated_transcription = transcriber.get_updated_transcription()
        print("\nTranscription with Updated Speaker Names:")
        print(updated_transcription)

        # 3. show again the transcription with the updated names.
        final_transcription = transcriber.get_updated_transcription()
        print("\nFinal transcription with updated Speaker Names:")
        print(final_transcription)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
