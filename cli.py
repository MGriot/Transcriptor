import os
import json
import argparse
import concurrent.futures
from audio_transcriber import (
    AudioTranscriber,
    DEFAULT_OUTPUT_DIR,
    AUDIO_FILE_EXTENSIONS,
    WHISPER_MODELS,
    VAD_METHODS,
    DEFAULT_NUM_PROCESSES,
)

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
    num_processes=1,
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
        num_processes=num_processes,
    )


def get_interactive_args():
    """Get arguments through interactive CLI prompts."""
    args = {}
    config = load_config()

    print("\n=== Audio Transcription Interactive Setup ===\n")

    # Get audio input
    while True:
        audio_input = input("Enter path to audio file or directory: ").strip()
        if os.path.exists(audio_input):
            if os.path.isdir(audio_input):
                audio_files = [
                    f
                    for f in os.listdir(audio_input)
                    if f.lower().endswith(AUDIO_FILE_EXTENSIONS)
                ]

                if not audio_files:
                    print("No audio files found in the specified directory.")
                    continue

                print("\nAvailable audio files:")
                for i, file in enumerate(audio_files, 1):
                    print(f"{i}. {file}")

                while True:
                    selection = (
                        input("\nEnter file numbers (e.g., 1,3,4 or 'all'): ")
                        .strip()
                        .lower()
                    )
                    if selection == "all":
                        selected_files = audio_files
                        break

                    try:
                        indices = [int(x.strip()) - 1 for x in selection.split(",")]
                        if all(0 <= i < len(audio_files) for i in indices):
                            selected_files = [audio_files[i] for i in indices]
                            break
                        else:
                            print("Invalid file number(s). Please try again.")
                    except ValueError:
                        print(
                            "Invalid input. Please enter numbers separated by commas or 'all'."
                        )

                args["audio_input"] = audio_input
                args["selected_files"] = selected_files
            else:
                args["audio_input"] = audio_input
                args["selected_files"] = None
            break
        print("Invalid path. Please enter a valid file or directory path.")

    # Ask about diarization
    diarization_response = input("\nUse speaker diarization? [Y/n]: ").strip().lower()
    args["skip_diarization"] = not (
        diarization_response == "" or diarization_response.startswith("y")
    )

    # Get Hugging Face token if using diarization
    if not args["skip_diarization"]:
        if config.get("hf_token"):
            args["hf_token"] = config["hf_token"]
            print("\nUsing Hugging Face token from config file")
        else:
            token = input(
                "\nEnter Hugging Face API token (press Enter to skip): "
            ).strip()
            if token:
                args["hf_token"] = token
                # Ask if user wants to save token
                save_token = (
                    input("Do you want to save this token for future use? [y/N]: ")
                    .strip()
                    .lower()
                )
                if save_token.startswith("y"):
                    config["hf_token"] = token
                    save_config(config)
                    print("Token saved to config file")
            else:
                args["hf_token"] = None
    else:
        args["hf_token"] = None

    # Choose Whisper model
    print("\nAvailable Whisper models:", ", ".join(WHISPER_MODELS))
    while True:
        model = input(f"Choose Whisper model (default: base): ").strip().lower()
        model = model or "base"  # set default if empty
        if model in [m.lower() for m in WHISPER_MODELS]:
            args["whisper_model"] = model
            break
        print("Invalid model. Please choose from the available models.")

    # Language selection
    args["language"] = (
        input(
            "\nEnter language code (e.g., en, fr, es) or press Enter for auto-detection: "
        )
        .strip()
        .lower()
        or None
    )

    # VAD options
    vad_response = (
        input("\nUse Voice Activity Detection (VAD)? [y/N]: ").strip().lower()
    )
    args["use_vad"] = vad_response.startswith("y")

    if args["use_vad"]:
        print("Available VAD methods:", ", ".join(VAD_METHODS))
        while True:
            method = input("Choose VAD method (default: silero): ").strip().lower()
            method = method or "silero"
            if method in [m.lower() for m in VAD_METHODS]:
                args["vad_method"] = method
                break
            print("Invalid method. Please choose from the available methods.")
    else:
        args["vad_method"] = None

    # Advanced options
    print("\n=== Advanced Options ===")
    args["min_speakers"] = None
    args["max_speakers"] = None
    args["no_rename_speakers"] = False

    if not args["skip_diarization"]:
        print("\nSpeaker Configuration:")
        min_speakers = input(
            "Minimum number of speakers (press Enter to skip): "
        ).strip()
        args["min_speakers"] = int(min_speakers) if min_speakers.isdigit() else None

        max_speakers = input(
            "Maximum number of speakers (press Enter to skip): "
        ).strip()
        args["max_speakers"] = int(max_speakers) if max_speakers.isdigit() else None

        rename_response = (
            input("Enable interactive speaker renaming? [Y/n]: ").strip().lower()
        )
        args["no_rename_speakers"] = not (
            rename_response == "" or rename_response.startswith("y")
        )

    args["verbose"] = (
        input("\nEnable verbose output? [y/N]: ").strip().lower().startswith("y")
    )
    args["plot_word_alignment"] = (
        input("Enable word alignment plotting? [y/N]: ").strip().lower().startswith("y")
    )
    args["detect_disfluencies"] = (
        input("Enable disfluency detection? [y/N]: ").strip().lower().startswith("y")
    )

    # Parallel processing
    cores = os.cpu_count()
    print(f"\nYour system has {cores} CPU cores available.")
    while True:
        num_proc = input(
            f"Enter number of processes to use (1-{cores}, -1 for all cores, default: 1): "
        ).strip()
        if not num_proc:
            args["num_processes"] = 1
            break
        try:
            num_proc = int(num_proc)
            if num_proc == -1 or (1 <= num_proc <= cores):
                args["num_processes"] = num_proc
                break
            print(f"Please enter a number between 1 and {cores}, or -1")
        except ValueError:
            print("Please enter a valid number")

    # Output directory
    args["output_dir"] = (
        input("\nEnter output directory path (press Enter for 'output'): ").strip()
        or DEFAULT_OUTPUT_DIR
    )

    print("\n=== Configuration Complete ===")
    return args


def main():
    # Check if any command-line arguments were provided
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="Transcribe audio files with speaker diarization and/or VAD."
        )
        parser.add_argument(
            "audio_input",
            help=(
                "Path to an audio file or a directory containing audio files. If a "
                "directory is provided, the script will process all audio files in "
                "that directory."
            ),
        )
        parser.add_argument(
            "--output_dir",
            default=DEFAULT_OUTPUT_DIR,
            help="Directory where output files will be saved. Defaults to 'output'.",
        )
        parser.add_argument(
            "--hf_token",
            default=None,
            help="Hugging Face API token for speaker diarization.",
        )
        parser.add_argument(
            "--skip_diarization",
            action="store_true",
            help="Skip speaker diarization and transcribe the entire audio file.",
        )
        parser.add_argument(
            "--whisper_model",
            default="base",
            choices=WHISPER_MODELS,
            help="Choose Whisper model size. Defaults to 'base'.",
        )
        parser.add_argument(
            "--language",
            default=None,
            help="Specify audio language (e.g., 'en', 'fr'). Auto-detects if not specified.",
        )
        parser.add_argument(
            "--min_speakers",
            type=int,
            default=None,
            help="Minimum number of speakers expected in the audio.",
        )
        parser.add_argument(
            "--max_speakers",
            type=int,
            default=None,
            help="Maximum number of speakers expected in the audio.",
        )
        parser.add_argument(
            "--use_vad", action="store_true", help="Enable voice activity detection."
        )
        parser.add_argument(
            "--vad_method",
            default=None,
            choices=VAD_METHODS,
            help="Choose VAD method. Defaults to 'silero' if --use_vad is set.",
        )
        parser.add_argument(
            "--verbose", action="store_true", help="Enable verbose output."
        )
        parser.add_argument(
            "--plot_word_alignment",
            action="store_true",
            help="Enable word alignment plotting.",
        )
        parser.add_argument(
            "--detect_disfluencies",
            action="store_true",
            help="Enable disfluency detection.",
        )
        parser.add_argument(
            "--no_rename_speakers",
            action="store_true",
            help="Disable interactive speaker renaming.",
        )
        parser.add_argument(
            "--num_processes",
            type=int,
            default=DEFAULT_NUM_PROCESSES,
            help="Number of processes for parallel processing. Use -1 for all CPU cores.",
        )
        parser.add_argument(
            "--selected_files",
            nargs="+",
            default=None,
            help="List of specific files to process when input is a directory",
        )

        args = parser.parse_args()

        # Initialize optional attributes that might be missing
        if not hasattr(args, "min_speakers"):
            args.min_speakers = None
        if not hasattr(args, "max_speakers"):
            args.max_speakers = None
        if not hasattr(args, "selected_files"):
            args.selected_files = None
        if not hasattr(args, "no_rename_speakers"):
            args.no_rename_speakers = False
    else:
        # If no arguments provided, use interactive mode
        args = argparse.Namespace(**get_interactive_args())

    audio_input = args.audio_input
    num_processes = args.num_processes

    if num_processes == -1:
        num_processes = os.cpu_count()
        print(f"Using all available CPU cores: {num_processes}")
    elif num_processes < 1:
        print("Number of processes must be >= 1. Using 1 process.")
        num_processes = 1

    if os.path.isdir(audio_input):
        audio_files = args.selected_files or [
            f
            for f in os.listdir(audio_input)
            if f.lower().endswith(AUDIO_FILE_EXTENSIONS)
        ]
        if not audio_files:
            print("No audio files found in the specified directory.")
            return

        print(
            f"Processing {len(audio_files)} audio files with {num_processes} processes."
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_processes
        ) as executor:
            futures = [
                executor.submit(
                    process_single_audio,
                    os.path.join(audio_input, audio_file),
                    args.output_dir,
                    args.hf_token,
                    args.skip_diarization,
                    args.whisper_model,
                    args.language,
                    args.min_speakers,
                    args.max_speakers,
                    args.use_vad,
                    args.vad_method,
                    args.verbose,
                    args.plot_word_alignment,
                    args.detect_disfluencies,
                    args.no_rename_speakers,
                    num_processes,
                )
                for audio_file in audio_files
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")
        print("Finished processing all audio files.")

    elif os.path.isfile(audio_input):
        print(f"Processing single audio file: {audio_input}")
        process_single_audio(
            audio_input,
            args.output_dir,
            args.hf_token,
            args.skip_diarization,
            args.whisper_model,
            args.language,
            args.min_speakers,
            args.max_speakers,
            args.use_vad,
            args.vad_method,
            args.verbose,
            args.plot_word_alignment,
            args.detect_disfluencies,
            args.no_rename_speakers,
            num_processes,
        )
        print("Finished processing.")
    else:
        print("Invalid audio input. Please provide a valid file or directory.")


if __name__ == "__main__":
    import sys

    main()
