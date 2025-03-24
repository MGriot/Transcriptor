# https://github.com/openai/whisper
# https://github.com/linto-ai/whisper-timestamped

import whisper
import logging
import os
import torch
from pyannote.audio import Pipeline
import soundfile as sf
import numpy as np

# Set up logging for any warnings or errors
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class AudioTranscriber:
    """
    A class to transcribe audio files using OpenAI's Whisper model with speaker diarization.
    """

    def __init__(
        self,
        model_name="base",
        diarization_model="pyannote/speaker-diarization@2023.11.16",
    ):
        """
        Initializes the AudioTranscriber object with a pre-loaded Whisper model and a diarization pipeline.

        Args:
            model_name (str, optional): The name of the Whisper model to use (e.g., "tiny", "base", "small", "medium", "large").
                Defaults to "base". Larger models are more accurate but require more resources.
            diarization_model (str, optional): The name of the pyannote.audio pipeline to use for speaker diarization.
                Defaults to "pyannote/speaker-diarization@2023.11.16".
        """
        try:
            # Try to load the specified Whisper model.
            self.model = whisper.load_model(model_name)
            self.model_name = model_name  # Save the model name
            logging.info(f"Whisper model '{model_name}' loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load Whisper model '{model_name}': {e}")
            raise

        try:
            # Initialize the pyannote.audio pipeline for speaker diarization.
            self.pipeline = Pipeline.from_pretrained(diarization_model)
            self.pipeline.to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            self.diarization_model = diarization_model
            logging.info(
                f"Diarization pipeline '{diarization_model}' loaded successfully."
            )
        except Exception as e:
            logging.error(
                f"Failed to load diarization pipeline '{diarization_model}': {e}"
            )
            raise

    def transcribe_audio(self, file_audio, language=None, auto_detect_language=False):
        """
        Transcribes the provided audio file, automatically detecting the language or using the provided one,
        and performs speaker diarization.

        Args:
            file_audio (str): The path to the audio file to transcribe.
            language (str, optional): The language to use for transcription (e.g., "it", "en", "fr").
                If None, the language is automatically detected. Defaults to None.
            auto_detect_language (bool, optional): Whether to detect the language automatically. Defaults to False.

        Returns:
            dict: A dictionary containing the transcription text and the detected language, with speaker labels.
            For example: {"text": "transcribed text", "language": "en", "segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Hello"}, ...]}
        Raises:
            FileNotFoundError: If the audio file does not exist.
            Exception: If an error occurs during transcription or diarization.
        """
        # Verify that the audio file exists.
        import os

        if not os.path.exists(file_audio):
            logging.error(f"Audio file not found: {file_audio}")
            raise FileNotFoundError(f"Audio file not found: {file_audio}")

        try:
            # 1. Load the audio file
            logging.info(f"Loading audio file: {file_audio}")
            audio, sample_rate = sf.read(file_audio)

            # 2. Perform language detection if auto_detect_language is True
            if auto_detect_language:
                logging.info("Auto-detecting language...")
                mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
                with torch.no_grad():
                    _, probs = self.model.detect_language(mel)
                detected_language = max(probs, key=probs.get)
                logging.info(f"Detected language: {detected_language}")
            elif language:
                detected_language = language
                logging.info(f"Using provided language: {language}")
            else:
                detected_language = "en"  # Default language if none is provided
                logging.info("Using default language: en")

            # 3. Perform transcription using Whisper
            logging.info(f"Transcribing audio with language: {detected_language}")
            # Check for CPU and use fp16=False
            if self.model.device == torch.device("cpu"):
                transcription = self.model.transcribe(
                    file_audio, language=detected_language, fp16=False
                )
            else:
                transcription = self.model.transcribe(
                    file_audio, language=detected_language
                )
            transcribed_text = transcription["text"]

            # 4. Perform speaker diarization using pyannote.audio
            logging.info("Performing speaker diarization...")
            try:
                # Convert the audio data to the format expected by pyannote.audio
                if (
                    len(audio.shape) > 1
                ):  # If audio is multi-channel, take the first channel
                    audio_data = audio[:, 0]
                else:
                    audio_data = audio
                diarization = self.pipeline(
                    {"waveform": torch.tensor(audio_data), "sample_rate": sample_rate}
                )
                logging.info("Speaker diarization completed.")
            except Exception as e:
                logging.error(f"Error during diarization: {e}")
                raise Exception(f"Error during speaker diarization: {e}") from e

            # 5. Combine transcription and diarization results
            logging.info("Combining transcription and diarization results...")
            segments = []
            for segment in transcription["segments"]:
                start_time = segment["start"]
                end_time = segment["end"]
                # Find the speaker label for the segment's time range
                speaker = None
                for dia_segment in diarization.itertracks(yield_label=True):
                    if (
                        dia_segment[0].start <= start_time
                        and dia_segment[0].end >= end_time
                    ):
                        speaker = dia_segment[2]
                        break
                if speaker:
                    segments.append(
                        {
                            "start": start_time,
                            "end": end_time,
                            "speaker": speaker,
                            "text": segment["text"],
                        }
                    )
                else:
                    segments.append(
                        {
                            "start": start_time,
                            "end": end_time,
                            "speaker": "UNKNOWN",
                            "text": segment["text"],
                        }
                    )

            logging.info(f"Detected language: {detected_language}")
            logging.info("Transcription and diarization completed.")
            return {
                "text": transcribed_text,
                "language": detected_language,
                "segments": segments,
            }
        except Exception as e:
            # Catch any exception during transcription, log the error, and re-raise.
            logging.error(f"Error during audio file processing: {e}")
            raise Exception(f"Error during audio file processing: {e}") from e


def main():
    """
    Main function to execute the transcription and diarization of a sample audio file.
    """
    # Example usage of the AudioTranscriber class:
    sample_audio_file = r"C:\Users\Admin\Documents\Coding\Transcriptor\audio\Botanicario - Ribes Nero.wav"  # Replace with your audio file path
    # Create an empty audio file if it does not exist
    if not os.path.exists(sample_audio_file):
        with open(sample_audio_file, "w") as f:
            pass
        print(
            f"Created empty sample audio file: {sample_audio_file}. Replace it with a real audio file to test the transcription."
        )

    # Create an instance of the AudioTranscriber class with the "base" model.
    transcriber = AudioTranscriber(model_name="base")  # You can change the model here

    try:
        # Perform transcription without specifying the language (automatic detection).
        result = transcriber.transcribe_audio(
            sample_audio_file, auto_detect_language=True
        )
        print("\nTranscription and Diarization (automatic language detection):")
        print(f"Detected language: {result['language']}")
        for segment in result["segments"]:
            print(
                f"[{segment['start']:02f} - {segment['end']:02f}] {segment['speaker']}: {segment['text']}"
            )

        # Perform transcription forcing the Italian language.
        result = transcriber.transcribe_audio(sample_audio_file, language="it")
        print(f"\nTranscription and Diarization (forced language: Italian):")
        print(f"Detected language: {result['language']}")
        for segment in result["segments"]:
            print(
                f"[{segment['start']:02f} - {segment['end']:02f}] {segment['speaker']}: {segment['text']}"
            )

        # Perform transcription forcing the English language
        result = transcriber.transcribe_audio(sample_audio_file, language="en")
        print(f"\nTranscription and Diarization (forced language: English):")
        print(f"Detected language: {result['language']}")
        for segment in result["segments"]:
            print(
                f"[{segment['start']:02f} - {segment['end']:02f}] {segment['speaker']}: {segment['text']}"
            )
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
