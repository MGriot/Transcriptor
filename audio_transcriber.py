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
import concurrent.futures
import tqdm
import traceback
from exceptions import *
import sys

# Constants for default values
DEFAULT_OUTPUT_DIR = "output"
AUDIO_FILE_EXTENSIONS = (".wav", ".mp3", ".ogg", ".flac", ".aac")
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
VAD_METHODS = ["silero", "silero:3.1", "auditok"]
DEFAULT_NUM_PROCESSES = 1


def _process_chunk(args):
    """Helper function to process a single chunk (for multiprocessing)."""
    chunk, language, vad_option, verbose, plot_word_alignment, detect_disfluencies = (
        args
    )
    logging.info(f"Transcribing {chunk['file_path']}")

    try:
        audio_dir = os.path.dirname(os.path.dirname(chunk["file_path"]))
        transcriber = AudioTranscriber(chunk["file_path"], output_dir=audio_dir)
        transcription, lang = transcriber.transcribe_chunk(
            chunk["file_path"],
            language=language,
            vad=vad_option,
            verbose=verbose,
            plot_word_alignment=plot_word_alignment,
            detect_disfluencies=detect_disfluencies,
        )
        return (
            {**chunk, "transcription": transcription, "language": lang}
            if transcription
            else None
        )
    except Exception as e:
        logging.error(f"Error processing chunk {chunk['file_path']}: {e}")
        return None


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
        self.audio_base_name = os.path.splitext(os.path.basename(audio_file))[0]
        self.output_dir_base = output_dir
        self.output_dir = os.path.join(output_dir, self.audio_base_name)
        self.chunks_dir = os.path.join(self.output_dir, "chunks")
        self.hf_token = hf_token
        self.whisper_model = whisper_model
        self.skip_diarization = skip_diarization
        self._init_directories()

    def _init_directories(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def load_audio(self, audio_file_path: str) -> Tuple[torch.Tensor, int]:
        """Loads an audio file using torchaudio."""
        try:
            if not os.path.exists(audio_file_path):
                raise AudioFileError(f"Audio file not found: {audio_file_path}")

            extension = os.path.splitext(audio_file_path)[1].lower()
            if extension not in AUDIO_FILE_EXTENSIONS:
                raise AudioFileError(f"Unsupported audio format: {extension}")

            waveform, sample_rate = torchaudio.load(audio_file_path)
            return waveform, sample_rate
        except Exception as e:
            logging.error(f"Error loading audio: {str(e)}\n{traceback.format_exc()}")
            raise AudioFileError(f"Failed to load audio: {str(e)}")

    def run_diarization(
        self, max_speakers: Optional[int] = None, min_speakers: Optional[int] = None
    ) -> Tuple[Pipeline, Annotation, Dict[str, str]]:
        """Performs speaker diarization."""
        try:
            if not self.hf_token:
                raise TokenError("HuggingFace token required for diarization")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=self.hf_token
            ).to(device)

            waveform, sample_rate = self.load_audio(self.audio_file)
            diarization = pipeline(
                {
                    "waveform": waveform,
                    "sample_rate": sample_rate,
                    **({"max_speakers": max_speakers} if max_speakers else {}),
                    **({"min_speakers": min_speakers} if min_speakers else {}),
                }
            )

            speaker_labels = {
                label for _, _, label in diarization.itertracks(yield_label=True)
            }
            speaker_names = {
                label: f"Speaker {i+1}"
                for i, label in enumerate(sorted(speaker_labels))
            }
            return pipeline, diarization, speaker_names
        except Exception as e:
            logging.error(f"Diarization error: {str(e)}\n{traceback.format_exc()}")
            raise DiarizationError(f"Diarization failed: {str(e)}")

    def chunk_audio(self, diarization: Annotation) -> List[Dict]:
        """Chunks audio based on diarization."""
        if not os.path.exists(self.chunks_dir):
            os.makedirs(self.chunks_dir, exist_ok=True)

        for old_file in os.listdir(self.chunks_dir):
            if old_file.startswith("chunk_") and old_file.endswith(".mp3"):
                try:
                    os.remove(os.path.join(self.chunks_dir, old_file))
                except OSError:
                    pass

        audio = AudioSegment.from_file(self.audio_file)
        chunks = []
        total_chunks = sum(1 for _ in diarization.itertracks(yield_label=True))

        print("\nSplitting audio into chunks:")
        for i, (turn, _, speaker) in enumerate(
            tqdm.tqdm(
                diarization.itertracks(yield_label=True),
                total=total_chunks,
                desc="Chunking",
                unit="segment",
            ),
            1,
        ):
            start_ms, end_ms = int(turn.start * 1000), int(turn.end * 1000)
            chunk_path = os.path.join(self.chunks_dir, f"chunk_{i}.mp3")
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
        vad: Optional[bool or str] = None,
        verbose: bool = False,
        plot_word_alignment: bool = False,
        detect_disfluencies: bool = False,
    ) -> Tuple[Dict, Optional[str]]:
        """Transcribes an audio chunk."""
        try:
            model = whisper.load_model(self.whisper_model)
            audio = whisper.load_audio(audio_file_path)
            result = whisper.transcribe(
                model,
                audio,
                language=language,
                vad=vad if vad else None,
                verbose=verbose,
                plot_word_alignment=plot_word_alignment,
                detect_disfluencies=detect_disfluencies,
            )
            return result, result.get("language") if language is None else language
        except Exception as e:
            logging.error(f"Transcription error: {str(e)}\n{traceback.format_exc()}")
            raise TranscriptionError(f"Chunk transcription failed: {str(e)}")

    def process_and_transcribe_chunks(
        self,
        chunks: List[Dict],
        language: Optional[str] = None,
        use_vad: bool = False,
        vad_method: Optional[str] = None,
        verbose: bool = False,
        plot_word_alignment: bool = False,
        detect_disfluencies: bool = False,
        num_processes: int = 1,
    ) -> Tuple[List[Dict], Optional[str]]:
        """Processes and saves transcriptions for individual chunks."""
        transcriptions = []
        detected_language = None
        vad_option = vad_method if use_vad else None

        chunk_args = [
            (
                chunk,
                language,
                vad_option,
                verbose,
                plot_word_alignment,
                detect_disfluencies,
            )
            for chunk in chunks
        ]

        print(f"\nProcessing {len(chunks)} audio chunks:")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_processes
        ) as executor:
            results = list(
                tqdm.tqdm(
                    executor.map(_process_chunk, chunk_args),
                    total=len(chunk_args),
                    desc="Transcribing",
                    unit="chunk",
                )
            )

        for result in results:
            if result:
                transcriptions.append(result)
                if detected_language is None and result.get("language"):
                    detected_language = result["language"]

        transcriptions_json_path = os.path.join(
            self.output_dir, f"{self.audio_base_name}_transcriptions.json"
        )
        with open(transcriptions_json_path, "w") as f:
            json.dump(transcriptions, f, indent=4)

        return transcriptions, detected_language

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
        num_processes: int = 1,
        no_rename_speakers: bool = False,  # Added parameter
    ):
        """Orchestrates the audio processing pipeline."""
        try:
            if self.skip_diarization:
                print("Transcribing whole audio file...")
                whole_transcription, detected_language = self.transcribe_chunk(
                    self.audio_file,
                    language=language,
                    vad=vad_method if use_vad else None,
                    verbose=verbose,
                    plot_word_alignment=plot_word_alignment,
                    detect_disfluencies=detect_disfluencies,
                )
                self._save_whole_transcription(whole_transcription, detected_language)
            else:
                _, diarization_result, speaker_names = self.run_diarization(
                    max_speakers, min_speakers
                )
                self._save_speaker_names(speaker_names)

                chunk_info_list = self.chunk_audio(diarization_result)
                transcriptions, detected_language = self.process_and_transcribe_chunks(
                    chunk_info_list,
                    language=language,
                    use_vad=use_vad,
                    vad_method=vad_method,
                    verbose=verbose,
                    plot_word_alignment=plot_word_alignment,
                    detect_disfluencies=detect_disfluencies,
                    num_processes=num_processes,
                )
                self._save_diarized_transcription(transcriptions, speaker_names)

        except Exception as e:
            logging.error(f"Processing error: {str(e)}\n{traceback.format_exc()}")
            if os.path.exists(self.chunks_dir):
                try:
                    for f in os.listdir(self.chunks_dir):
                        os.remove(os.path.join(self.chunks_dir, f))
                    os.rmdir(self.chunks_dir)
                except OSError:
                    pass
            raise TranscriptionError(f"Audio processing failed: {str(e)}")

    def _save_speaker_names(self, speaker_names: Dict[str, str]):
        """Save speaker names to JSON file."""
        speaker_names_path = os.path.join(
            self.output_dir, f"{self.audio_base_name}_speaker_names.json"
        )
        with open(speaker_names_path, "w") as f:
            json.dump(speaker_names, f, indent=4)
        print(f"Speaker names saved to {speaker_names_path}")

    def _save_whole_transcription(self, transcription: Dict, language: Optional[str]):
        """Save whole audio transcription."""
        cleaned = self._clean_whole_transcription(transcription, language)
        self._save_output_file(cleaned, "whole_transcription.txt")

    def _save_diarized_transcription(
        self, transcriptions: List[Dict], speaker_names: Dict[str, str]
    ):
        """Save diarized transcription."""
        cleaned = self._clean_diarized_transcription(transcriptions, speaker_names)
        self._save_output_file(cleaned, "diarized_transcription.txt")

    def _clean_whole_transcription(
        self, transcription: Dict, language: Optional[str]
    ) -> List[str]:
        """Clean whole audio transcription."""
        cleaned = []
        if language := (transcription.get("language") or language):
            cleaned.extend([f"Detected Language: {language.upper()}", ""])
        cleaned.extend([f"Audio File: {self.audio_base_name}", ""])
        cleaned.extend(
            f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}"
            for seg in transcription.get("segments", [])
        )
        return cleaned

    def _clean_diarized_transcription(
        self, transcriptions: List[Dict], speaker_names: Dict[str, str]
    ) -> List[str]:
        """Clean diarized transcription with timestamps."""
        cleaned = []
        if transcriptions:
            if lang := transcriptions[0].get("language"):
                cleaned.extend([f"Detected Language: {lang.upper()}", ""])
        cleaned.extend([f"Audio File: {self.audio_base_name}", ""])

        for chunk in transcriptions:
            speaker_label = chunk["speaker"]
            speaker_name = speaker_names.get(speaker_label, speaker_label)
            start = chunk["start_time"]
            end = chunk["end_time"]

            chunk_text = " ".join(
                seg["text"] for seg in chunk["transcription"].get("segments", [])
            ).strip()

            if chunk_text:
                cleaned.append(
                    f"{speaker_name} [{start:.2f} - {end:.2f}]: {chunk_text}"
                )

        return cleaned

    def _save_output_file(self, content: List[str], filename: str):
        """Save output to text file."""
        output_path = os.path.join(
            self.output_dir, f"{self.audio_base_name}_{filename}"
        )
        with open(output_path, "w") as f:
            f.write("\n".join(content))
        print(f"Transcription saved to {output_path}")


class SpeakerRenamer:
    def __init__(self, audio_base_name: str, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.audio_base_name = audio_base_name
        self.output_dir = output_dir  # Use the exact output directory path
        self.speaker_names_path = os.path.join(
            output_dir, f"{audio_base_name}_speaker_names.json"
        )
        self.transcriptions_path = os.path.join(
            output_dir, f"{audio_base_name}_transcriptions.json"
        )
        self.speaker_names = self._load_speaker_names()

    def _load_speaker_names(self) -> Dict[str, str]:
        """Load existing speaker names from JSON file"""
        if os.path.exists(self.speaker_names_path):
            with open(self.speaker_names_path, "r") as f:
                return json.load(f)
        return {}

    def rename_speakers_interactive(self):
        """Interactive speaker renaming process"""
        print("\nCurrent speaker names:")
        for label, name in self.speaker_names.items():
            print(f"{label}: {name}")

        if sys.stdin.isatty():
            try:
                new_names = {}
                for label, current_name in self.speaker_names.items():
                    new_name = input(
                        f"\nEnter new name for {current_name} ({label}) [Enter to keep]: "
                    ).strip()
                    new_names[label] = new_name or current_name
                self.speaker_names = new_names
                self._save_speaker_names()
            except EOFError:
                print("\nNon-interactive mode detected, keeping names")
        else:
            print("Non-interactive mode, keeping existing names")
        return self.speaker_names

    def generate_updated_transcriptions(self):
        """Generate new transcriptions with updated speaker names and timestamps."""
        with open(self.transcriptions_path, "r") as f:
            transcriptions = json.load(f)

        # Create cleaned text version with timestamps
        cleaned = []
        for chunk in transcriptions:
            speaker_label = chunk["speaker"]
            speaker_name = self.speaker_names.get(speaker_label, speaker_label)
            start = chunk["start_time"]
            end = chunk["end_time"]

            chunk_text = " ".join(
                seg["text"] for seg in chunk["transcription"].get("segments", [])
            ).strip()

            if chunk_text:
                cleaned.append(
                    f"{speaker_name} [{start:.2f} - {end:.2f}]: {chunk_text}"
                )

        # Save updated files
        updated_json_path = os.path.join(
            self.output_dir, f"{self.audio_base_name}_transcriptions_renamed.json"
        )
        with open(updated_json_path, "w") as f:
            json.dump(transcriptions, f, indent=4)

        txt_path = os.path.join(
            self.output_dir, f"{self.audio_base_name}_transcription_renamed.txt"
        )
        with open(txt_path, "w") as f:
            f.write("\n".join(cleaned))

        return updated_json_path, txt_path

    def _save_speaker_names(self):
        """Save current speaker names to JSON file"""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.speaker_names_path), exist_ok=True)
            with open(self.speaker_names_path, "w") as f:
                json.dump(self.speaker_names, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving speaker names: {e}")
            raise
