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

# Constants for default values
DEFAULT_OUTPUT_DIR = "output"
AUDIO_FILE_EXTENSIONS = (".wav", ".mp3", ".ogg", ".flac", ".aac")
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
VAD_METHODS = ["silero", "silero:3.1", "auditok"]
DEFAULT_NUM_PROCESSES = 1


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
        self.output_dir_base = output_dir
        self.output_dir = os.path.join(self.output_dir_base, self.audio_base_name)
        self.chunks_dir = os.path.join(
            self.output_dir_base, self.audio_base_name, "chunks"
        )
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
        num_processes: int = 1,  # Added num_processes
    ) -> List[Dict]:
        """Processes and saves transcriptions for individual chunks, using multiprocessing."""
        transcriptions = []
        detected_language = None

        def process_chunk(chunk):
            nonlocal detected_language
            logging.info(f"Transcribing {chunk['file_path']}")
            vad_option = vad_method if use_vad else None
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
                result = {**chunk, "transcription": transcription, "language": lang}
                if self.debug_mode:
                    print(f"Transcription for {chunk['file_path']} (Language: {lang}):")
                    for segment in transcription["segments"]:
                        print(
                            f"    [{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}"
                        )
                return result
            return None

        if num_processes > 1:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_processes
            ) as executor:
                futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        transcriptions.append(result)
        else:
            for chunk in chunks:
                result = process_chunk(chunk)
                if result:
                    transcriptions.append(result)

        transcriptions_json_path = os.path.join(
            self.output_dir, f"{self.audio_base_name}_transcriptions.json"
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
        vad_option = vad_method if use_vad else None
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
                    cleaned.append("")
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
            self.output_dir, f"{self.audio_base_name}_{filename}"
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
        num_processes: int = 1,  # Added num_processes parameter
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
                        self.output_dir, f"{self.audio_base_name}_speaker_names.json"
                    )
                    with open(speaker_names_path, "w") as f:
                        json.dump(speaker_names, f, indent=4)
                    print(f"Speaker names saved to {speaker_names_path}")

                (
                    transcriptions,
                    detected_language,
                ) = self.process_and_transcribe_chunks(
                    chunk_info_list,
                    language=language,
                    use_vad=use_vad,
                    vad_method=vad_method,
                    verbose=verbose,
                    plot_word_alignment=plot_word_alignment,
                    detect_disfluencies=detect_disfluencies,
                    num_processes=num_processes,  # Pass num_processes
                )
                transcriptions_json_path = os.path.join(
                    self.output_dir, f"{self.audio_base_name}_transcriptions.json"
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
