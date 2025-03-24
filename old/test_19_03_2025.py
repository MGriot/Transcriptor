# Description: Test file for the 19th of March 2025
# voglio creare una funzione o classe in python per fare la trascrizione di file audio mediante la libreria Whisper: # https://github.com/openai/whisper
# Vorrei che lo script accettasse in input un file audio e restituisse la trascrizione del file audio.
# Vorrei che riconoscesse in automatico la linga parlata ma che possa anche essere passata come variabile.
# https://gemini.google.com/app/bd8806cd36c22854?hl=en
import whisper
import logging
import os
import torch

# Set up logging for any warnings or errors
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class AudioTranscriber:
    """
    A class to transcribe audio files using OpenAI's Whisper model.
    """

    def __init__(self, model_name="base"):
        """
        Initializes the AudioTranscriber object with a pre-loaded Whisper model.

        Args:
            model_name (str, optional): The name of the Whisper model to use (e.g., "tiny", "base", "small", "medium", "large").
                Defaults to "base". Larger models are more accurate but require more resources.
        """
        try:
            # Try to load the specified model. If it fails, catch the exception.
            self.model = whisper.load_model(model_name)
            self.model_name = model_name  # save the model name
            logging.info(f"Whisper model '{model_name}' loaded successfully.")
        except Exception as e:
            # Log the error and re-raise the exception to inform the caller.
            logging.error(f"Failed to load Whisper model '{model_name}': {e}")
            raise

    def transcribe_audio(self, file_audio, language=None):
        """
        Transcribes the provided audio file, automatically detecting the language or using the provided one.

        Args:
            file_audio (str): The path to the audio file to transcribe.
            language (str, optional): The language to use for transcription (e.g., "it", "en", "fr").
                If None, the language is automatically detected. Defaults to None.

        Returns:
            str: The transcription of the audio file.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            Exception: If an error occurs during transcription.
        """
        # Verify that the audio file exists.
        import os

        if not os.path.exists(file_audio):
            logging.error(f"Audio file not found: {file_audio}")
            raise FileNotFoundError(f"Audio file not found: {file_audio}")

        try:
            # Perform the transcription with the specified language (if provided).
            if language:
                logging.info(
                    f"Transcribing '{file_audio}' with forced language: {language}"
                )
                transcription = self.model.transcribe(file_audio, language=language)
            else:
                logging.info(
                    f"Transcribing '{file_audio}' with automatic language detection."
                )
                # Check if the device is CPU and force FP32 if necessary
                if self.model.device == torch.device("cpu"):
                    transcription = self.model.transcribe(
                        file_audio, fp16=False
                    )  # Force FP32
                else:
                    transcription = self.model.transcribe(file_audio)
            # The transcription is in dictionary format; extract the text.
            transcribed_text = transcription["text"]
            logging.info("Transcription completed.")
            return transcribed_text
        except Exception as e:
            # Catch any exception during transcription, log the error, and re-raise.
            logging.error(f"Error during transcription of '{file_audio}': {e}")
            raise Exception(f"Error during audio file transcription: {e}") from e


def main():
    """
    Main function to execute the transcription of a sample audio file.
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
        automatic_transcription = transcriber.transcribe_audio(sample_audio_file)
        print("\nTranscription (automatic language detection):")
        print(automatic_transcription)

        # Perform transcription forcing the Italian language.
        italian_transcription = transcriber.transcribe_audio(
            sample_audio_file, language="it"
        )
        print("\nTranscription (forced language: Italian):")
        print(italian_transcription)

        # Perform transcription forcing the English language
        english_transcription = transcriber.transcribe_audio(
            sample_audio_file, language="en"
        )
        print("\nTranscription (forced language: English):")
        print(english_transcription)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
