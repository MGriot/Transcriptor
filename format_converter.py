import os
import logging
from moviepy import VideoFileClip, AudioFileClip  # Updated import style
from exceptions import AudioFileError

# Supported input formats (including video)
SUPPORTED_FORMATS = {
    "audio": (".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".aif", ".aiff"),
    "video": (".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"),
}


def convert_to_compatible_format(input_path: str, target_format: str = "wav") -> str:
    """
    Convert any supported audio/video file to a compatible audio format.
    Returns path to the converted file.
    """
    try:
        input_ext = os.path.splitext(input_path)[1].lower()
        if input_ext == f".{target_format}":
            return input_path

        output_path = os.path.splitext(input_path)[0] + f"_converted.{target_format}"

        # Handle video files
        if input_ext in SUPPORTED_FORMATS["video"]:
            logging.info(f"Converting video file: {input_path}")
            with VideoFileClip(input_path) as video:
                audio = video.audio
                if audio:
                    with audio as audio_clip:  # Context manager for audio
                        audio_clip.write_audiofile(output_path)
                else:
                    raise AudioFileError(f"Video file {input_path} has no audio track.")
            return output_path

        # Handle audio files
        elif input_ext in SUPPORTED_FORMATS["audio"]:
            logging.info(f"Converting audio file: {input_path}")
            with AudioFileClip(
                input_path
            ) as audio:  # Directly use MoviePy's AudioFileClip
                audio.write_audiofile(output_path)
            return output_path

        else:
            raise AudioFileError(f"Unsupported file format: {input_ext}")

    except Exception as e:
        raise AudioFileError(f"Error converting file {input_path}: {str(e)}")


def cleanup_converted_file(file_path: str):
    """Clean up temporary converted files."""
    if "_converted." in file_path:
        try:
            os.remove(file_path)
            logging.info(f"Cleaned up converted file: {file_path}")
        except OSError as e:
            logging.warning(f"Could not remove temporary file {file_path}: {e}")
