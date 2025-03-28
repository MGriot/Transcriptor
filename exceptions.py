class TranscriptionError(Exception):
    """Base exception for transcription errors"""

    pass


class AudioFileError(TranscriptionError):
    """Raised when there are issues with audio files"""

    pass


class DiarizationError(TranscriptionError):
    """Raised when there are issues with speaker diarization"""

    pass


class ConfigurationError(TranscriptionError):
    """Raised when there are configuration issues"""

    pass


class VADError(TranscriptionError):
    """Raised when there are VAD-related issues"""

    pass


class TokenError(TranscriptionError):
    """Raised when there are HuggingFace token issues"""

    pass
