import logging
import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from typing import List, Dict, Tuple, Optional
import os
import json
from pydub import AudioSegment
import whisper_timestamped as whisper
import argparse
import sys  # Import the sys module


# Constants for default values
DEFAULT_OUTPUT_DIR = "output"
AUDIO_FILE_EXTENSIONS = (".wav", ".mp3", ".ogg", ".flac", ".aac")  # Add more if needed
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
VAD_METHODS = ["silero", "silero:3.1", "auditok"]

class AudioTranscriber:
    def __init__(
        self,
        audio_file: str,
        output_dir: str = "output",
        hf_token: Optional[str] = None,
        skip_diarization: bool = False,
        whisper_model: str = "base",
    ):
        self.audio_file = audio_file
        self.audio_base_name = os.path.splitext(os.path.basename(self.audio_file))[0]
        self.output_dir_base = output_dir  # Base output directory
        self.output_dir = os.path.join(
            self.output_dir_base, self.audio_base_name
        )  # final output dir for json and txt
        self.chunks_dir = os.path.join(
            self.output_dir_base, self.audio_base_name, "chunks"
        ) # directory for audio chunks
        self.hf_token = hf_token
        self.whisper_model = whisper_model
        self.debug_mode = False
        self.skip_diarization = skip_diarization
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.chunks_dir):
            os.makedirs(self.chunks_dir)

    def load_audio(self, audio_file_path: str) -> Tuple[torch.Tensor, int]:
        """Loads an audio file using torchaudio."""
        try:
            waveform, sample_rate = torchaudio.load(audio_file_path)
            return waveform, sample_rate
        except Exception as e:
            logging.error(f"Error loading audio file: {e}")
            raise

    def run_diarization(
        self,
        max_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
    ) -> Tuple[Pipeline, Annotation, Dict[str, str]]:
        """Performs speaker diarization."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=self.hf_token
        ).to(device)
        waveform, sample_rate = self.load_audio(self.audio_file)
        input_data = {"waveform": waveform, "sample_rate": sample_rate}
        if max_speakers:
            input_data["max_speakers"] = max_speakers
        if min_speakers:
            input_data["min_speakers"] = min_speakers
        diarization = pipeline(input_data)
        speaker_labels = set()
        for segment, _, label in diarization.itertracks(yield_label=True):
            speaker_labels.add(label)
        speaker_names = {
            label: f"Speaker {i + 1}" for i, label in enumerate(sorted(speaker_labels))
        }
        return pipeline, diarization, speaker_names

    def chunk_audio(self, diarization: Annotation) -> List[Dict]:
        """Chunks audio based on diarization."""
        audio = AudioSegment.from_file(self.audio_file)
        chunks = []
        for i, (turn, _, speaker) in enumerate(
            diarization.itertracks(yield_label=True), 1
        ):
            start_ms, end_ms = int(turn.start * 1000), int(turn.end * 1000)
            chunk_path = os.path.join(self.chunks_dir, f"chunk_{i}.mp3") # save chunks in chunks dir
            audio[start_ms:end_ms].export(chunk_path, format="mp3")
            chunks.append(
                {
                    "file_path": chunk_path,
                    "speaker": speaker,
                    "start_time": turn.start,
                    "end_time": turn.end,
                }
            )
        return chunks

    def transcribe_chunk(
        self,
        audio_file_path: str,
        language: Optional[str] = None,
        vad: Optional[bool or str or List[Tuple[float, float]]] = None,
        verbose: bool = False,
        plot_word_alignment: bool = False,
        detect_disfluencies: bool = False,
    ) -> Tuple[Dict, Optional[str]]:
        """Transcribes an audio chunk and returns the transcription and detected language."""
        try:
            model = whisper.load_model(self.whisper_model)
            audio = whisper.load_audio(audio_file_path)

            if vad is not None and vad is not False:
                logging.info(
                    f"Performing voice activity detection with settings: {vad}"
                )
                if vad is True or vad == "silero":
                    result = whisper.transcribe(
                        model,
                        audio,
                        language=language,
                        vad="silero",
                        verbose=verbose,
                        plot_word_alignment=plot_word_alignment,
                        detect_disfluencies=detect_disfluencies,
                    )
                elif vad == "silero:3.1":
                    result = whisper.transcribe(
                        model,
                        audio,
                        language=language,
                        vad="silero:3.1",
                        verbose=verbose,
                        plot_word_alignment=plot_word_alignment,
                        detect_disfluencies=detect_disfluencies,
                    )
                elif vad == "auditok":
                    result = whisper.transcribe(
                        model,
                        audio,
                        language=language,
                        vad="auditok",
                        verbose=verbose,
                        plot_word_alignment=plot_word_alignment,
                        detect_disfluencies=detect_disfluencies,
                    )
                elif isinstance(vad, list):
                    speech_segments = []
                    for start, end in vad:
                        speech_segments.append(
                            audio[
                                int(start * whisper.audio.SAMPLE_RATE) : int(
                                    end * whisper.audio.SAMPLE_RATE
                                )
                            ]
                        )
                    if speech_segments:
                        full_transcription = {"segments": []}
                        for segment in speech_segments:
                            segment_result = whisper.transcribe(
                                model,
                                segment,
                                language=language,
                                verbose=verbose,
                                plot_word_alignment=plot_word_alignment,
                                detect_disfluencies=detect_disfluencies,
                            )
                            full_transcription["segments"].extend(
                                segment_result.get(
                                    "segments",
                                )
                            )
                        result = full_transcription
                else:
                    logging.warning(
                        f"Invalid VAD setting: {vad}. Transcribing without VAD."
                    )
                    result = whisper.transcribe(
                        model,
                        audio,
                        language=language,
                        verbose=verbose,
                        plot_word_alignment=plot_word_alignment,
                        detect_disfluencies=detect_disfluencies,
                    )
            else:
                result = whisper.transcribe(
                    model,
                    audio,
                    language=language,
                    verbose=verbose,
                    plot_word_alignment=plot_word_alignment,
                    detect_disfluencies=detect_disfluencies,
                )

            detected_language = result.get("language") if language is None else language
            return result, detected_language
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return {}, None

    def process_and_transcribe_chunks(
        self,
        chunks: List[Dict],
        language: Optional[str] = None,
        use_vad: bool = False,
        vad_method: Optional[str] = None,
        verbose: bool = False,
        plot_word_alignment: bool = False,
        detect_disfluencies: bool = False,
    ) -> List[Dict]:
        """Processes and saves transcriptions for individual chunks."""
        transcriptions = []
        detected_language = None
        for chunk in chunks:
            logging.info(f"Transcribing {chunk['file_path']}")
            vad_option = None
            if use_vad:
                vad_option = (
                    vad_method if vad_method else True
                )  # Use default if no method specified
            transcription, lang = self.transcribe_chunk(
                chunk["file_path"],
                language=language,
                vad=vad_option,
                verbose=verbose,
                plot_word_alignment=plot_word_alignment,
                detect_disfluencies=detect_disfluencies,
            )
            if transcription:
                if detected_language is None and lang is not None:
                    detected_language = lang
                transcriptions.append(
                    {**chunk, "transcription": transcription, "language": lang}
                )
                if self.debug_mode:  # only print if debug mode is on
                    print(f"Transcription for {chunk['file_path']} (Language: {lang}):")
                    for segment in transcription["segments"]:
                        print(
                            f"    [{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}"
                        )

        transcriptions_json_path = os.path.join(
            self.output_dir, f"{self.audio_base_name}_transcriptions.json" # save json in output dir
        )
        with open(transcriptions_json_path, "w") as f:
            json.dump(transcriptions, f, indent=4)
        return transcriptions, detected_language

    def transcribe_whole_audio(
        self,
        language: Optional[str] = None,
        use_vad: bool = False,
        vad_method: Optional[str] = None,
        verbose: bool = False,
        plot_word_alignment: bool = False,
        detect_disfluencies: bool = False,
    ) -> Tuple[Dict, Optional[str]]:
        """Transcribes the entire audio file without diarization."""
        logging.info(f"Transcribing the entire audio file: {self.audio_file}")
        vad_option = None
        if use_vad:
            vad_option = vad_method if vad_method else True
        return self.transcribe_chunk(
            self.audio_file,
            language=language,
            vad=vad_option,
            verbose=verbose,
            plot_word_alignment=plot_word_alignment,
            detect_disfluencies=detect_disfluencies,
        )

    def clean_transcription(
        self, transcriptions_json: str, speaker_names: Dict[str, str]
    ) -> List[str]:
        """Cleans the transcription JSON to a readable format (for diarized audio)."""
        with open(transcriptions_json, "r") as f:
            transcriptions = json.load(f)

        cleaned = []
        detected_language = (
            transcriptions[0].get("language") if transcriptions else None
        )
        if detected_language:
            cleaned.append(f"Detected Language: {detected_language.upper()}")
            cleaned.append("")
        cleaned.append(f"Audio File: {self.audio_base_name}")
        cleaned.append("")

        current_speaker, current_text, current_start, current_end = None, "", None, None
        for i, chunk in enumerate(transcriptions):
            if current_speaker != chunk["speaker"]:
                if current_speaker is not None:
                    cleaned.append(
                        f"{speaker_names.get(current_speaker, current_speaker)} [{current_start:.2f} - {current_end:.2f}]: {current_text}"
                    )
                current_speaker, current_text = chunk["speaker"], ""
                current_start, current_end = chunk["start_time"], chunk["end_time"]
                if (
                    i > 0
                    and transcriptions[i]["speaker"] != transcriptions[i - 1]["speaker"]
                ):
                    cleaned.append(
                        ""
                    )  # Add a blank line before a new speaker (after the first)
            if chunk["transcription"] and chunk["transcription"]["segments"]:
                current_text += " ".join(
                    seg["text"] for seg in chunk["transcription"]["segments"]
                )
        if current_speaker:
            cleaned.append(
                f"{speaker_names.get(current_speaker, current_speaker)} [{current_start:.2f} - {current_end:.2f}]: {current_text}"
            )
        return [line for line in cleaned if line.strip() != ""]

    def clean_whole_transcription(
        self, whole_transcription: Dict, language: Optional[str] = None
    ) -> List[str]:
        """Cleans the whole transcription output to a readable format (without diarization)."""
        cleaned = []
        detected_language = (
            whole_transcription.get("language") if whole_transcription else language
        )
        if detected_language:
            cleaned.append(f"Detected Language: {detected_language.upper()}")
            cleaned.append("")
        cleaned.append(f"Audio File: {self.audio_base_name}")
        cleaned.append("")
        if whole_transcription and whole_transcription.get("segments"):
            for segment in whole_transcription["segments"]:
                cleaned.append(
                    f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}"
                )
        return cleaned

    def save_transcription_to_file(
        self, cleaned_transcriptions: List[str], filename="transcription.txt"
    ):
        """Saves the cleaned transcription to a text file."""
        output_file = os.path.join(
            self.output_dir, f"{self.audio_base_name}_{filename}" # save txt in output dir
        )
        with open(output_file, "w") as f:
            f.write("\n".join(cleaned_transcriptions))

    def process_audio(
        self,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        use_vad: bool = False,
        vad_method: Optional[str] = None,
        verbose: bool = False,
        plot_word_alignment: bool = False,
        detect_disfluencies: bool = False,
        no_rename_speakers: bool = False,
    ):
        """Orchestrates the audio processing pipeline."""
        try:
            if self.skip_diarization:
                print("Skipping diarization and transcribing the whole audio file.")
                whole_transcription, detected_language = self.transcribe_whole_audio(
                    language=language,
                    use_vad=use_vad,
                    vad_method=vad_method,
                    verbose=verbose,
                    plot_word_alignment=plot_word_alignment,
                    detect_disfluencies=detect_disfluencies,
                )
                if whole_transcription:
                    cleaned_transcriptions = self.clean_whole_transcription(
                        whole_transcription, detected_language
                    )
                    print("\nTranscription:")
                    for line in cleaned_transcriptions:
                        print(line)
                    self.save_transcription_to_file(
                        cleaned_transcriptions, filename="whole_transcription.txt"
                    )
                else:
                    print("Transcription failed.")
            else:
                pipeline, diarization_result, speaker_names = self.run_diarization(
                    max_speakers=max_speakers, min_speakers=min_speakers
                )
                chunk_info_list = self.chunk_audio(diarization_result)
                print("Audio file diarized and chunked.")

                num_speakers = len(speaker_names)
                print(f"\nDetected {num_speakers} speakers:")
                for i, (turn, _, speaker) in enumerate(
                    diarization_result.itertracks(yield_label=True), 1
                ):
                    print(
                        f"Chunk {i}: Speaker '{speaker}' [{turn.start:.2f} - {turn.end:.2f}]"
                    )

                # Speaker renaming moved here
                if not no_rename_speakers:
                    new_speaker_names = {}
                    for label in sorted(speaker_names.keys()):
                        new_name = input(
                            f"Enter a new name for '{speaker_names[label]}' (default: {speaker_names[label]}): "
                        ).strip()
                        if new_name:
                            new_speaker_names[label] = new_name
                        else:
                            new_speaker_names[label] = speaker_names[label]
                    speaker_names.update(new_speaker_names)

                    speaker_names_path = os.path.join(
                        self.output_dir, f"{self.audio_base_name}_speaker_names.json" #save speaker names in output dir
                    )
                    with open(speaker_names_path, "w") as f:
                        json.dump(speaker_names, f, indent=4)
                    print(f"Speaker names saved to {speaker_names_path}")

                transcriptions, detected_language = self.process_and_transcribe_chunks(
                    chunk_info_list,
                    language=language,
                    use_vad=use_vad,
                    vad_method=vad_method,
                    verbose=verbose,
                    plot_word_alignment=plot_word_alignment,
                    detect_disfluencies=detect_disfluencies,
                )
                transcriptions_json_path = os.path.join(
                    self.output_dir, f"{self.audio_base_name}_transcriptions.json" # get path of json
                )
                cleaned_transcriptions = self.clean_transcription(
                    transcriptions_json_path, speaker_names
                )
                print("Cleaned Transcriptions:")
                for paragraph in cleaned_transcriptions:
                    print(paragraph)

                self.save_transcription_to_file(
                    cleaned_transcriptions, filename="diarized_transcription.txt"
                )

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            print(f"An error occurred: {e}")


def process_single_audio(
    audio_file_path,
    output_dir="output",
    hf_token=None,
    skip_diarization=False,
    whisper_model="base",
    language=None,
    min_speakers=None,
    max_speakers=None,
    use_vad=False,
    vad_method=None,
    verbose=False,
    plot_word_alignment=False,
    detect_disfluencies=False,
    no_rename_speakers=False,
):
    transcriber = AudioTranscriber(
        audio_file_path,
        output_dir,
        hf_token,
        skip_diarization=skip_diarization,
        whisper_model=whisper_model,
    )
    transcriber.process_audio(
        language=language,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        use_vad=use_vad,
        vad_method=vad_method,
        verbose=verbose,
        plot_word_alignment=plot_word_alignment,
        detect_disfluencies=detect_disfluencies,
        no_rename_speakers=no_rename_speakers,
    )



def main():
    """Main function to parse arguments and process audio."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files with speaker diarization and/or VAD."
    )
    parser.add_argument(
        "audio_input",
        help=(
            "Path to an audio file or a directory containing audio files.  If a "
            "directory is provided, the script will process all audio files in "
            "that directory."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "The main directory where all output files will be saved.  Defaults to "
            "'output'.  If the input is a directory, a subdirectory with the name "
            "of each audio file will be created inside this directory."
        ),
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help=(
            "Hugging Face API token.  Required for speaker diarization.  If not "
            "provided, the script will attempt to read it from a config.py file."
        ),
    )
    parser.add_argument(
        "--skip_diarization",
        action="store_true",
        help=(
            "Skip speaker diarization and transcribe the entire audio file as a "
            "single speaker."
        ),
    )
    parser.add_argument(
        "--whisper_model",
        default="base",
        choices=WHISPER_MODELS,
        help="Choose a Whisper model size.  Defaults to 'base'.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help=(
            "Specify the language of the audio file (e.g., 'en', 'fr', 'es').  If "
            "not provided, Whisper will attempt to detect the language."
        ),
    )
    parser.add_argument(
        "--min_speakers",
        type=int,
        default=None,
        help="Minimum number of speakers expected in the audio. Used for diarization.",
    )
    parser.add_argument(
        "--max_speakers",
        type=int,
        default=None,
        help="Maximum number of speakers expected in the audio. Used for diarization.",
    )
    parser.add_argument(
        "--use_vad",
        action="store_true",
        help=(
            "Enable voice activity detection to remove silent parts before "
            "transcription."
        ),
    )
    parser.add_argument(
        "--vad_method",
        default=None,
        choices=VAD_METHODS,
        help=(
            "Choose a VAD method: 'silero',  'silero:3.1', or 'auditok'.  If "
            "--use_vad is set and this is not provided, 'silero' is used as "
            "default."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for Whisper.",
    )
    parser.add_argument(
        "--plot_word_alignment",
        action="store_true",
        help="Enable plotting word alignment (if supported by the model).",
    )
    parser.add_argument(
        "--detect_disfluencies",
        action="store_true",
        help="Enable disfluency detection (if supported by the model).",
    )
    parser.add_argument(
        "--no_rename_speakers",
        action="store_true",
        help="Disable the interactive prompt to rename speakers.",
    )

    if len(sys.argv) == 1:
        # Interactive mode
        output_dir_base = (
            input("Enter the main output directory (default: output): ").strip()
            or "output"
        )

        # Load HF_TOKEN if config.py exists
        hf_token = None
        try:
            from config import HF_TOKEN as config_token

            hf_token = config_token
        except ImportError:
            logging.warning(
                "config.py not found or HF_TOKEN not defined. Diarization may not work."
                "  Please create a config.py file with HF_TOKEN = 'YOUR_HUGGINGFACE_TOKEN'"
            )

        audio_input = input(
            "Enter the path to an audio file or a directory containing audio files: "
        ).strip()

        if os.path.isdir(audio_input):
            audio_files = [
                f for f in os.listdir(audio_input) if f.endswith(AUDIO_FILE_EXTENSIONS)
            ]
            if not audio_files:
                print("No audio files found in the specified directory.")
                return

            print("\nAvailable audio files:")
            for i, filename in enumerate(audio_files):
                print(f"{i + 1}. {filename}")

            while True:
                selection = (
                    input(
                        "\nChoose files to process (e.g., 'all', '1', '2,3,4', '1-3'): "
                    )
.lower()
                    .strip()
                )
                files_to_process = []

                if selection == "all":
                    files_to_process = [
                        os.path.join(audio_input, f) for f in audio_files
                    ]
                    break
                elif "," in selection:
                    indices = [s.strip() for s in selection.split(",")]
                    valid_indices = True
                    selected_indices = set()
                    for index_str in indices:
                        if index_str.isdigit():
                            index = int(index_str)
                            if 1 <= index <= len(audio_files):
                                selected_indices.add(index - 1)
                            else:
                                print(f"Invalid file number: {index_str}")
                                valid_indices = False
                                break
                        else:
                            print(f"Invalid input: {index_str}")
                            valid_indices = False
                            break
                    if valid_indices:
                        files_to_process = [
                            os.path.join(audio_input, audio_files[i])
                            for i in sorted(list(selected_indices))
                        ]
                        break
                elif "-" in selection:
                    try:
                        start_str, end_str = selection.split("-")
                        start_index = int(start_str.strip())
                        end_index = int(end_str.strip())
                        if (
                            1 <= start_index <= len(audio_files)
                            and 1 <= end_index <= len(audio_files)
                            and start_index <= end_index
                        ):
                            files_to_process = [
                                os.path.join(audio_input, audio_files[i])
                                for i in range(start_index - 1, end_index)
                            ]
                            break
                        else:
                            print("Invalid range of file numbers.")
                    except ValueError:
                        print("Invalid range format.")
                elif selection.isdigit():
                    index = int(selection)
                    if 1 <= index <= len(audio_files):
                        files_to_process = [
                            os.path.join(audio_input, audio_files[index - 1])
                        ]
                        break
                    else:
                        print("Invalid file number.")
                else:
                    print(
                        "Invalid selection format. Please use 'all', a single number, comma-separated numbers, or a range (e.g., '1-3')."
                    )

            for audio_file_path in files_to_process:
                print(f"\n--- Processing: {audio_file_path} ---")
                # Ask for processing options for each file
                skip_diarization_input = input(
                    "Skip speaker diarization for this file? (yes/no): "
                ).lower()
                skip_diarization = skip_diarization_input == "yes"

                whisper_model_choice = "base"
                min_speakers = None  # set default values
                max_speakers = None
                if not skip_diarization:
                    specify_speakers = input(
                        "Specify min/max speakers for this file? (yes/no): "
                    ).lower()

                    if specify_speakers == "yes":
                        try:
                            min_speakers = int(
                                input(
                                    "Enter the minimum number of speakers (optional): "
                                )
                                or None
                            )
                            max_speakers = int(
                                input(
                                    "Enter the maximum number of speakers (optional): "
                                )
                                or None
                            )
                        except ValueError:
                            print("Invalid input for the number of speakers.")

                print("\nAvailable Whisper models: tiny, base, small, medium, large")
                chosen_model = (
                    input("Choose a Whisper model (default: base): ").lower().strip()
                )
                if chosen_model in ["tiny", "base", "small", "medium", "large"]:
                    whisper_model_choice = chosen_model
                elif chosen_model:
                    print(f"Invalid model '{chosen_model}'. Using default 'base'.")

                specify_language = input(
                    "Specify a language for this file? (yes/no): "
                ).lower()
                transcription_language = None
                if specify_language == "yes":
                    transcription_language = input(
                        "Enter the language code (e.g., en, fr, es): "
                    ).strip()
                use_vad_input = input(
                    "Use Voice Activity Detection (VAD)? (yes/no): "
                ).lower()
                use_vad = use_vad_input == "yes"
                vad_method_choice = None
                if use_vad:
                    print(f"Available VAD methods: {', '.join(VAD_METHODS)}")
                    vad_method_choice = (
                        input("Choose a VAD method (default: silero): ").lower().strip()
                    )
                    if vad_method_choice not in VAD_METHODS:
                        print("Invalid VAD method. Using default 'silero'")
                        vad_method_choice = "silero"

                verbose_output = input("Enable verbose output? (yes/no): ").lower()
                verbose = verbose_output == "yes"

                plot_alignment_input = input("Plot word alignment? (yes/no): ").lower()
                plot_word_alignment = plot_alignment_input == "yes"

                detect_disfluencies_input = input(
                    "Detect disfluencies? (yes/no): "
                ).lower()
                detect_disfluencies = detect_disfluencies_input == "yes"

                no_rename_speakers_input = input(
                    "Disable speaker renaming prompt? (yes/no): "
                ).lower()
                no_rename_speakers = no_rename_speakers_input == "yes"

                process_single_audio(
                    audio_file_path,
                    os.path.join(output_dir_base, "chunks"),
                    hf_token,
                    skip_diarization,
                    whisper_model_choice,
                    transcription_language,
                    min_speakers,
                    max_speakers,
                    use_vad,
                    vad_method_choice,
                    verbose,
                    plot_word_alignment,
                    detect_disfluencies,
                    no_rename_speakers,
                )

        elif os.path.isfile(audio_input):
            # Process a single audio file
            output_dir_base = (
                input("Enter the output directory (default: output): ").strip()
                or "output"
            )
            skip_diarization = (
                input("Skip diarization? (yes/no): ").lower().strip() == "yes"
            )
            whisper_model_choice = (
                input("Choose a Whisper model (default: base): ").lower().strip()
                or "base"
            )
            if whisper_model_choice not in WHISPER_MODELS:
                print(f"Invalid model '{whisper_model_choice}'. Using default 'base'.")
                whisper_model_choice = "base"

            language_input = (
                input("Specify language (e.g., 'en', default: None): ").strip() or None
            )

            min_speakers = None
            max_speakers = None
            if not skip_diarization:
                specify_speakers = input(
                    "Specify min/max speakers? (yes/no): "
                ).lower()
                if specify_speakers == "yes":
                    try:
                        min_speakers = int(
                            input("Enter min speakers (optional): ").strip() or None
                        )
                        max_speakers = int(
                            input("Enter max speakers (optional): ").strip() or None
                        )
                    except ValueError:
                        print("Invalid input for number of speakers. Skipping...")
                        min_speakers = None
                        max_speakers = None

            use_vad_input = input("Use VAD? (yes/no): ").lower().strip()
            use_vad = use_vad_input == "yes"
            vad_method_choice = None
            if use_vad:
                vad_method_choice = (
                    input("Choose VAD method (default: silero): ").lower().strip()
                    or "silero"
                )
                if vad_method_choice not in VAD_METHODS:
                    print("Invalid VAD method. Using default 'silero'.")
                    vad_method_choice = "silero"

            verbose_output = input("Enable verbose output? (yes/no): ").lower().strip()
            verbose = verbose_output == "yes"

            plot_alignment_input = input("Plot word alignment? (yes/no): ").lower().strip()
            plot_word_alignment = plot_alignment_input == "yes"

            detect_disfluencies_input = input(
                "Detect disfluencies? (yes/no): "
            ).lower().strip()
            detect_disfluencies = detect_disfluencies_input == "yes"

            no_rename_speakers_input = input(
                "Disable speaker renaming prompt? (yes/no): "
            ).lower().strip()
            no_rename_speakers = no_rename_speakers_input == "yes"
            process_single_audio(
                audio_input,
                os.path.join(output_dir_base, "chunks"),
                hf_token,
                skip_diarization,
                whisper_model_choice,
                language_input,
                min_speakers,
                max_speakers,
                use_vad,
                vad_method_choice,
                verbose,
                plot_word_alignment,
                detect_disfluencies,
                no_rename_speakers,
            )
        else:
            print("Invalid input. Please provide a valid audio file or directory.")
    else:
        # Command-line mode
        args = parser.parse_args()
        audio_input = args.audio_input
        output_dir_base = args.output_dir
        hf_token = args.hf_token
        skip_diarization = args.skip_diarization
        whisper_model_choice = args.whisper_model
        language_input = args.language
        min_speakers = args.min_speakers
        max_speakers = args.max_speakers
        use_vad = args.use_vad
        vad_method_choice = args.vad_method
        verbose = args.verbose
        plot_word_alignment = args.plot_word_alignment
        detect_disfluencies = args.detect_disfluencies
        no_rename_speakers = args.no_rename_speakers

        if os.path.isdir(audio_input):
            audio_files = [
                f for f in os.listdir(audio_input) if f.endswith(AUDIO_FILE_EXTENSIONS)
            ]
            if not audio_files:
                print("No audio files found in the specified directory.")
                return
            for audio_file_path in audio_files:
                print(f"\n--- Processing: {audio_file_path} ---")
                process_single_audio(
                    os.path.join(audio_input, audio_file_path),
                    os.path.join(output_dir_base, "chunks"),
                    hf_token,
                    skip_diarization,
                    whisper_model_choice,
                    language_input,
                    min_speakers,
                    max_speakers,
                    use_vad,
                    vad_method_choice,
                    verbose,
                    plot_word_alignment,
                    detect_disfluencies,
                    no_rename_speakers,
                )
        elif os.path.isfile(audio_input):
            process_single_audio(
                audio_input,
                os.path.join(output_dir_base, "chunks"),
                hf_token,
                skip_diarization,
                whisper_model_choice,
                language_input,
                min_speakers,
                max_speakers,
                use_vad,
                vad_method_choice,
                verbose,
                plot_word_alignment,
                detect_disfluencies,
                no_rename_speakers,
            )
        else:
            print("Invalid input. Please provide a valid audio file or directory.")



if __name__ == "__main__":
    main()
