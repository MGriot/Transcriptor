from typing import Optional
from exceptions import ConfigurationError


def validate_language_code(lang: Optional[str]) -> Optional[str]:
    """Validate language code format."""
    if lang is None:
        return None
    if not isinstance(lang, str) or len(lang) not in [2, 3]:
        raise ConfigurationError(
            f"Invalid language code: {lang}. Use ISO 639-1/2 format (e.g., 'en', 'fra')"
        )
    return lang.lower()


def validate_speakers(
    min_speakers: Optional[int], max_speakers: Optional[int]
) -> tuple:
    """Validate speaker count configuration."""
    if min_speakers is not None:
        if not isinstance(min_speakers, int) or min_speakers < 1:
            raise ConfigurationError("Minimum speakers must be a positive integer")

    if max_speakers is not None:
        if not isinstance(max_speakers, int) or max_speakers < 1:
            raise ConfigurationError("Maximum speakers must be a positive integer")

    if min_speakers and max_speakers and min_speakers > max_speakers:
        raise ConfigurationError(
            f"Minimum speakers ({min_speakers}) cannot be greater than maximum speakers ({max_speakers})"
        )

    return min_speakers, max_speakers


def validate_vad_config(
    use_vad: bool, vad_method: Optional[str], valid_methods: list
) -> tuple:
    """Validate VAD configuration."""
    if use_vad and not vad_method:
        vad_method = "silero"  # Default method

    if vad_method and vad_method not in valid_methods:
        raise ConfigurationError(
            f"Invalid VAD method: {vad_method}. Supported methods: {', '.join(valid_methods)}"
        )

    return use_vad, vad_method


def validate_processes(num_processes: int, max_cores: int) -> int:
    """Validate number of processes."""
    if num_processes == -1:
        return max_cores
    if not isinstance(num_processes, int) or num_processes < 1:
        raise ConfigurationError("Number of processes must be a positive integer or -1")
    if num_processes > max_cores:
        raise ConfigurationError(
            f"Number of processes ({num_processes}) exceeds available CPU cores ({max_cores})"
        )
    return num_processes
