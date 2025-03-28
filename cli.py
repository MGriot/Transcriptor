import os
import sys
import json
import logging
import argparse
import traceback
import concurrent.futures
import tqdm
from audio_transcriber import (
    AudioTranscriber,
    SpeakerRenamer,
    DEFAULT_OUTPUT_DIR,
    AUDIO_FILE_EXTENSIONS,
    WHISPER_MODELS,
    VAD_METHODS,
    DEFAULT_NUM_PROCESSES,
)
from exceptions import *
from validation import (
    validate_language_code,
    validate_speakers,
    validate_vad_config,
    validate_processes,
)
from logging_config import setup_logging

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")


def load_config():
    """Load configuration from JSON file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Warning: Config file is invalid, using defaults")
    return {}


def save_config(config_data):
    """Save configuration to JSON file."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config_data, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not save config file: {e}")


try:
    from config import HF_TOKEN as CONFIG_HF_TOKEN
except ImportError:
    CONFIG_HF_TOKEN = None


def get_operation_mode():
    """Get user's desired operation mode."""
    print("\n=== Main Menu ===")
    print("1. Process new audio transcription")
    print("2. Modify speaker names in existing transcription")
    print("3. Exit")

    while True:
        choice = input("\nChoose operation (1-3): ").strip()
        if choice in ["1", "2", "3"]:
            return choice
        print("Invalid choice. Please enter 1, 2, or 3")


def process_single_audio(audio_file_path, output_dir="output", **kwargs):
    """Process a single audio file."""
    try:
        if not os.path.exists(audio_file_path):
            raise AudioFileError(f"Input file not found: {audio_file_path}")

        os.makedirs(output_dir, exist_ok=True)

        transcriber = AudioTranscriber(
            audio_file=audio_file_path,
            output_dir=output_dir,
            hf_token=kwargs.get("hf_token"),
            skip_diarization=kwargs.get("skip_diarization", False),
            whisper_model=kwargs.get("whisper_model", "base"),
        )

        transcriber.process_audio(
            language=kwargs.get("language"),
            min_speakers=kwargs.get("min_speakers"),
            max_speakers=kwargs.get("max_speakers"),
            use_vad=kwargs.get("use_vad", False),
            vad_method=kwargs.get("vad_method"),
            verbose=kwargs.get("verbose", False),
            num_processes=kwargs.get("num_processes", 1),
        )

        audio_base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        print(f"\nTranscription completed for {audio_base_name}")
        print(f"Output files saved in: {os.path.join(output_dir, audio_base_name)}")

    except Exception as e:
        logging.error(f"Error processing {audio_file_path}: {str(e)}")
        raise


def find_transcription_files(base_dir: str = DEFAULT_OUTPUT_DIR) -> list:
    """Find all transcription files in output directory and subdirectories."""
    transcription_files = []
    try:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith("_transcriptions.json"):
                    audio_name = file.replace("_transcriptions.json", "")
                    rel_path = os.path.relpath(root, base_dir)
                    output_dir = root  # Store the actual output directory path
                    if rel_path == ".":
                        rel_path = ""
                    transcription_files.append((audio_name, rel_path, output_dir))
    except Exception as e:
        print(f"Error scanning directory: {e}")
    return transcription_files


def modify_existing_transcription():
    """Handle speaker name modification for existing transcription."""
    print("\n=== Modify Existing Transcription ===")

    output_dir = (
        input(
            f"\nEnter base output directory (default: {DEFAULT_OUTPUT_DIR}): "
        ).strip()
        or DEFAULT_OUTPUT_DIR
    )

    transcriptions = find_transcription_files(output_dir)

    if not transcriptions:
        print(f"\nNo transcription files found in {output_dir}")
        return

    print("\nAvailable transcriptions:")
    for i, (name, subdir, _) in enumerate(transcriptions, 1):
        path = os.path.join(subdir, name) if subdir else name
        print(f"{i}. {path}")

    while True:
        try:
            choice = (
                input("\nEnter number to modify (or 'q' to quit): ").strip().lower()
            )
            if choice == "q":
                return

            idx = int(choice) - 1
            if 0 <= idx < len(transcriptions):
                audio_name, _, actual_output_dir = transcriptions[idx]

                try:
                    renamer = SpeakerRenamer(audio_name, actual_output_dir)
                    print(f"\nModifying: {audio_name}")
                    print("Current speaker names:")
                    for label, name in renamer.speaker_names.items():
                        print(f"- {name} ({label})")

                    if renamer.rename_speakers_interactive():
                        json_path, txt_path = renamer.generate_updated_transcriptions()
                        print(f"\nUpdated files created:")
                        print(f"- JSON: {json_path}")
                        print(f"- Text: {txt_path}")
                except Exception as e:
                    print(f"\nError modifying speakers: {str(e)}")
                    logging.error(
                        f"Modification error: {str(e)}\n{traceback.format_exc()}"
                    )

                if (
                    input("\nModify another transcription? [y/N]: ").lower().strip()
                    != "y"
                ):
                    break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            print(f"Error: {e}")
            break


def select_files_interactive(directory_path):
    """Interactive file selection from directory."""
    audio_files = [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.lower().endswith(AUDIO_FILE_EXTENSIONS)
    ]

    if not audio_files:
        print("No audio files found in directory")
        return []

    print("\nFound audio files:")
    for i, file_path in enumerate(audio_files, 1):
        print(f"{i}. {os.path.basename(file_path)}")

    while True:
        selection = (
            input("\nEnter file numbers separated by commas or 'all' (q to quit): ")
            .strip()
            .lower()
        )
        if selection == "q":
            return []
        if selection == "all":
            return audio_files

        try:
            indices = [int(i.strip()) - 1 for i in selection.split(",")]
            selected = [audio_files[i] for i in indices if 0 <= i < len(audio_files)]
            return selected if selected else []
        except (ValueError, IndexError):
            print("Invalid input. Please enter numbers separated by commas or 'all'")


def process_directory(directory_path, output_dir, params):
    """Process multiple files in a directory."""
    # Use already selected files if available
    selected_files = params.pop("selected_files", None) or select_files_interactive(
        directory_path
    )
    if not selected_files:
        return

    print(f"\nProcessing {len(selected_files)} files:")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=params.get("num_processes", 1)
    ) as executor:
        futures = []
        for file_path in selected_files:
            futures.append(
                executor.submit(process_single_audio, file_path, output_dir, **params)
            )

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing files",
        ):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file: {str(e)}")


def get_interactive_args():
    """Get arguments through interactive prompts."""
    args = {}
    config = load_config()

    print("\n=== New Transcription Setup ===")

    # Path handling and immediate file selection
    while True:
        path = input("\nEnter path to audio file/directory: ").strip()
        if os.path.exists(path):
            if os.path.isdir(path):
                selected_files = select_files_interactive(path)
                if not selected_files:  # User quit or no files selected
                    continue
                args["selected_files"] = selected_files  # Store full paths
                args["path"] = path
            else:
                args["path"] = path
                args["selected_files"] = None
            break
        print("Invalid path. Please try again.")

    # Rest of configuration after file selection
    print("\n=== Processing Configuration ===")

    # Diarization configuration
    args["skip_diarization"] = not (
        input("\nUse speaker diarization? [Y/n]: ").strip().lower() in ["", "y", "yes"]
    )

    if not args["skip_diarization"]:
        args["hf_token"] = (
            config.get("hf_token")
            or input("\nEnter Hugging Face token (required): ").strip()
        )
        if not args["hf_token"]:
            raise TokenError("Hugging Face token required for diarization")

        if not config.get("hf_token"):
            if input("Save token for future? [y/N]: ").strip().lower().startswith("y"):
                config["hf_token"] = args["hf_token"]
                save_config(config)

    # Model selection
    print("\nAvailable Whisper models:", ", ".join(WHISPER_MODELS))
    args["whisper_model"] = "base"
    while True:
        model = input("Choose model (default: base): ").strip().lower()
        if model in {m.lower(): m for m in WHISPER_MODELS}:
            args["whisper_model"] = model
            break
        if not model:
            break

    # Language and VAD
    args["language"] = input("\nLanguage code (leave blank for auto): ").strip() or None
    args["use_vad"] = input("Use VAD? [y/N]: ").lower().startswith("y")
    args["vad_method"] = None
    if args["use_vad"]:
        print("Available VAD methods:", ", ".join(VAD_METHODS))
        args["vad_method"] = next(
            (
                m
                for m in VAD_METHODS
                if input("Choose method (default: silero): ").strip().lower()
                in [m.lower(), ""]
            ),
            "silero",
        )

    # Speaker limits
    args["min_speakers"], args["max_speakers"] = None, None
    if not args["skip_diarization"]:
        try:
            args["min_speakers"] = int(input("Min speakers (optional): ") or 0) or None
            args["max_speakers"] = int(input("Max speakers (optional): ") or 0) or None
        except ValueError:
            pass

    # Output and processing
    args["output_dir"] = (
        input(f"\nOutput directory (default: {DEFAULT_OUTPUT_DIR}): ")
        or DEFAULT_OUTPUT_DIR
    )
    args["num_processes"] = int(input("Parallel processes (default 1): ") or 1)
    args["verbose"] = input("Verbose output? [y/N]: ").lower().startswith("y")

    return args


def handle_command_line():
    """Handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Audio Transcription CLI")
    parser.add_argument("path", help="Audio file/directory path")
    parser.add_argument("-o", "--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hf-token", help="Hugging Face token")
    parser.add_argument("--whisper-model", choices=WHISPER_MODELS, default="base")
    parser.add_argument("--language", help="Language code")
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--use-vad", action="store_true")
    parser.add_argument("--vad-method", choices=VAD_METHODS)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--modify", action="store_true")
    parser.add_argument("--audio-base-name")
    return parser.parse_args()


def main():
    try:
        logger, _ = setup_logging()

        if len(sys.argv) > 1:
            args = handle_command_line()

            if args.modify:
                if not args.audio_base_name:
                    raise ValueError("--audio-base-name required for modification")
                renamer = SpeakerRenamer(args.audio_base_name, args.output_dir)
                renamer.rename_speakers_interactive()
                renamer.generate_updated_transcriptions()
                return

            params = {
                "hf_token": args.hf_token or CONFIG_HF_TOKEN,
                "skip_diarization": False,
                "whisper_model": args.whisper_model,
                "language": args.language,
                "min_speakers": args.min_speakers,
                "max_speakers": args.max_speakers,
                "use_vad": args.use_vad,
                "vad_method": args.vad_method,
                "num_processes": args.num_processes,
                "verbose": args.verbose,
            }

            if os.path.isdir(args.path):
                process_directory(args.path, args.output_dir, params)
            else:
                process_single_audio(args.path, args.output_dir, **params)
        else:
            while True:  # Main program loop
                choice = get_operation_mode()

                if choice == "3":  # Exit
                    print("\nGoodbye!")
                    break

                if choice == "1":  # New transcription
                    try:
                        cli_args = get_interactive_args()
                        if os.path.isdir(cli_args["path"]):
                            # Pass selected files to process_directory
                            process_directory(
                                cli_args["path"],
                                cli_args["output_dir"],
                                {
                                    k: v
                                    for k, v in cli_args.items()
                                    if k not in ["path", "output_dir"]
                                },
                            )
                        else:
                            process_single_audio(**cli_args)
                        input("\nPress Enter to return to main menu...")
                    except Exception as e:
                        logger.error(f"Processing error: {str(e)}", exc_info=True)
                        print(f"\nError: {str(e)}")
                        input("\nPress Enter to continue...")

                elif choice == "2":  # Modify speakers
                    try:
                        modify_existing_transcription()
                        input("\nPress Enter to return to main menu...")
                    except Exception as e:
                        logger.error(f"Modification error: {str(e)}", exc_info=True)
                        print(f"\nError: {str(e)}")
                        input("\nPress Enter to continue...")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
