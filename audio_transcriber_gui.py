# done with deepseeker R1 + search

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import queue
import sys
import threading
import os
import logging
from audio_transcriber import (
    AudioTranscriber,  # Changed from process_single_audio
    DEFAULT_OUTPUT_DIR,
    AUDIO_FILE_EXTENSIONS,
    VAD_METHODS,
)
from exceptions import *
from validation import (
    validate_language_code,
    validate_speakers,
    validate_vad_config,
    validate_processes,
)
from logging_config import setup_logging
from log_handler import ThreadSafeLogger, QueueHandler
import tkinter.messagebox as tk_messagebox


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Transcriber GUI")

        # Initialize logging
        self.logger = ThreadSafeLogger().get_logger()
        self.log_queue = ThreadSafeLogger().get_queue()
        self.queue_handler = ThreadSafeLogger()._instance.queue_handler

        # GUI Setup
        self.setup_gui()

        # Start log consumer thread
        self.running = True
        self.log_thread = threading.Thread(target=self.consume_logs, daemon=True)
        self.log_thread.start()

        self.processing = False  # Add flag for processing state

    def setup_gui(self):
        # Input Section
        input_frame = ttk.LabelFrame(self, text="Input")
        input_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        self.input_path = tk.StringVar()
        ttk.Label(input_frame, text="Audio File/Directory:").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(input_frame, textvariable=self.input_path, width=50).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Button(input_frame, text="Browse", command=self.browse_input).grid(
            row=0, column=2
        )

        # Output Directory
        self.output_dir = tk.StringVar(value=DEFAULT_OUTPUT_DIR)
        ttk.Label(input_frame, text="Output Directory:").grid(
            row=1, column=0, sticky="w"
        )
        ttk.Entry(input_frame, textvariable=self.output_dir, width=50).grid(
            row=1, column=1, sticky="ew"
        )
        ttk.Button(input_frame, text="Browse", command=self.browse_output).grid(
            row=1, column=2
        )

        # Parameters Section
        params_frame = ttk.LabelFrame(self, text="Processing Parameters")
        params_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        # Hugging Face Token
        ttk.Label(params_frame, text="Hugging Face Token:").grid(
            row=0, column=0, sticky="w"
        )
        self.hf_token = ttk.Entry(params_frame, width=50)
        self.hf_token.grid(row=0, column=1, columnspan=2, sticky="ew")

        # Diarization Options
        self.skip_diarization = tk.BooleanVar()
        ttk.Checkbutton(
            params_frame,
            text="Skip Speaker Diarization",
            variable=self.skip_diarization,
        ).grid(row=1, column=0, sticky="w")

        # Whisper Model
        ttk.Label(params_frame, text="Whisper Model:").grid(row=2, column=0, sticky="w")
        self.whisper_model = ttk.Combobox(
            params_frame, values=["tiny", "base", "small", "medium", "large"]
        )
        self.whisper_model.set("base")
        self.whisper_model.grid(row=2, column=1, sticky="w")

        # Language
        # Language
        ttk.Label(params_frame, text="Language Code:").grid(row=3, column=0, sticky="w")
        self.language = ttk.Entry(params_frame)
        self.language.grid(row=3, column=1, sticky="w")

        # Speaker Limits
        ttk.Label(params_frame, text="Min Speakers:").grid(row=4, column=0, sticky="w")
        self.min_speakers = ttk.Spinbox(params_frame, from_=1, to=10, width=5)
        self.min_speakers.grid(row=4, column=1, sticky="w")

        ttk.Label(params_frame, text="Max Speakers:").grid(row=5, column=0, sticky="w")
        self.max_speakers = ttk.Spinbox(params_frame, from_=1, to=10, width=5)
        self.max_speakers.grid(row=5, column=1, sticky="w")

        # VAD Options
        self.use_vad = tk.BooleanVar()
        ttk.Checkbutton(params_frame, text="Enable VAD", variable=self.use_vad).grid(
            row=6, column=0, sticky="w"
        )
        ttk.Label(params_frame, text="VAD Method:").grid(row=6, column=1, sticky="w")
        self.vad_method = ttk.Combobox(params_frame, values=VAD_METHODS)
        self.vad_method.grid(row=6, column=2, sticky="w")

        # Additional Options
        options_frame = ttk.LabelFrame(self, text="Additional Options")
        options_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        self.verbose = tk.BooleanVar()
        ttk.Checkbutton(
            options_frame, text="Verbose Output", variable=self.verbose
        ).grid(row=0, column=0, sticky="w")

        self.plot_word_alignment = tk.BooleanVar()
        ttk.Checkbutton(
            options_frame, text="Plot Word Alignment", variable=self.plot_word_alignment
        ).grid(row=0, column=1, sticky="w")

        self.detect_disfluencies = tk.BooleanVar()
        ttk.Checkbutton(
            options_frame, text="Detect Disfluencies", variable=self.detect_disfluencies
        ).grid(row=0, column=2, sticky="w")

        self.no_rename_speakers = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="Disable Speaker Renaming",
            variable=self.no_rename_speakers,
        ).grid(row=1, column=0, sticky="w")

        # Add Process Controls frame
        process_frame = ttk.LabelFrame(self, text="Processing Controls")
        process_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        # Add Start/Stop buttons
        self.start_button = ttk.Button(
            process_frame, text="Start Processing", command=self.start_processing
        )
        self.start_button.grid(row=0, column=0, padx=5, pady=5)

        self.stop_button = ttk.Button(
            process_frame,
            text="Stop Processing",
            command=self.stop_processing,
            state="disabled",
        )
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            process_frame, variable=self.progress_var, maximum=100, mode="determinate"
        )
        self.progress_bar.grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5
        )

        # Log Output below process controls
        self.log_area = scrolledtext.ScrolledText(self, state="disabled", height=15)
        self.log_area.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")

        # Configure grid weights
        self.grid_rowconfigure(4, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def setup_progress_bar(self):
        """Setup progress bar widget"""
        self.progress_frame = ttk.LabelFrame(self, text="Progress")
        self.progress_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

    def browse_input(self):
        """Browse for input file or directory"""
        is_file = tk.messagebox.askyesno(
            "Input Type", "Select a file? (No for directory)"
        )
        if is_file:
            path = filedialog.askopenfilename(
                title="Select Audio File",
                filetypes=[
                    (
                        "Audio Files",
                        " ".join(f"*{ext}" for ext in AUDIO_FILE_EXTENSIONS),
                    )
                ],
            )
        else:
            path = filedialog.askdirectory(title="Select Audio Directory")

        if path:
            self.input_path.set(path)

    def browse_output(self):
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_dir.set(path)

    def start_processing(self):
        """Start the transcription process"""
        if self.processing:
            return

        try:
            # Basic input validation
            if not self.input_path.get():
                raise ConfigurationError("Please select an input file or directory")

            if not self.skip_diarization.get() and not self.hf_token.get():
                if not tk_messagebox.askyesno(
                    "Missing Token",
                    "No Hugging Face token provided. This will disable speaker diarization. Continue?",
                ):
                    return

            # Update UI state
            self.processing = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.progress_var.set(0)

            # Get and validate parameters
            params = self.get_processing_parameters()

            # Start processing thread
            self.process_thread = threading.Thread(
                target=self.run_processing, args=(params,), daemon=True
            )
            self.process_thread.start()

            # Start progress update
            self.after(100, self.update_progress)

        except ConfigurationError as e:
            tk_messagebox.showerror("Configuration Error", str(e))
        except Exception as e:
            tk_messagebox.showerror("Error", f"Unexpected error: {str(e)}")
            self.logger.error(f"Error starting process: {str(e)}", exc_info=True)

    def stop_processing(self):
        """Stop the transcription process"""
        if not self.processing:
            return

        if tk_messagebox.askyesno("Confirm", "Stop current processing?"):
            self.processing = False
            self.logger.info("Stopping processing...")
            self.update_buttons_state()

    def run_processing(self, params):
        """Run the actual processing in a separate thread"""
        try:
            input_path = params["input_path"]
            if os.path.isdir(input_path):
                files = [
                    f
                    for f in os.listdir(input_path)
                    if f.lower().endswith(AUDIO_FILE_EXTENSIONS)
                ]
                total_files = len(files)

                for i, file in enumerate(files, 1):
                    if not self.processing:
                        self.logger.info("Processing stopped by user")
                        break

                    full_path = os.path.join(input_path, file)
                    try:
                        self.process_single_file(full_path, params)
                        self.progress_var.set((i / total_files) * 100)
                    except Exception as e:
                        self.logger.error(f"Error processing {file}: {str(e)}")
                        if not tk_messagebox.askyesno(
                            "Error",
                            f"Error processing {file}. Continue with remaining files?",
                        ):
                            break
            else:
                self.process_single_file(input_path, params)
                self.progress_var.set(100)

            self.logger.info("Processing completed")

        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}", exc_info=True)
            tk_messagebox.showerror("Error", f"Processing error: {str(e)}")
        finally:
            self.processing = False
            self.after(0, self.update_buttons_state)

    def update_buttons_state(self):
        """Update button states based on processing status"""
        self.start_button.config(state="normal" if not self.processing else "disabled")
        self.stop_button.config(state="disabled" if not self.processing else "normal")

    def update_progress(self):
        """Update progress bar and schedule next update if still processing"""
        if self.processing:
            self.after(100, self.update_progress)

    def get_processing_parameters(self):
        """Collect and validate all processing parameters"""
        try:
            language = self.language.get() or None
            language = validate_language_code(language)

            min_speakers = (
                int(self.min_speakers.get()) if self.min_speakers.get() else None
            )
            max_speakers = (
                int(self.max_speakers.get()) if self.max_speakers.get() else None
            )
            min_speakers, max_speakers = validate_speakers(min_speakers, max_speakers)

            use_vad, vad_method = validate_vad_config(
                self.use_vad.get(),
                self.vad_method.get() if self.vad_method.get() else None,
                VAD_METHODS,
            )

            return {
                "input_path": self.input_path.get(),
                "output_dir": self.output_dir.get() or DEFAULT_OUTPUT_DIR,
                "hf_token": self.hf_token.get() or None,
                "skip_diarization": self.skip_diarization.get(),
                "whisper_model": self.whisper_model.get() or "base",
                "language": language,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
                "use_vad": use_vad,
                "vad_method": vad_method,
                "verbose": self.verbose.get(),
                "plot_word_alignment": self.plot_word_alignment.get(),
                "detect_disfluencies": self.detect_disfluencies.get(),
                "no_rename_speakers": self.no_rename_speakers.get(),
            }
        except ValueError as e:
            raise ConfigurationError(f"Invalid parameter value: {str(e)}")

    def process_files(self, params):
        try:
            self.logger.info(f"Processing with parameters: {params}")
            if os.path.isdir(params["input_path"]):
                for file in os.listdir(params["input_path"]):
                    if file.lower().endswith(AUDIO_FILE_EXTENSIONS):
                        self.process_single_file(
                            os.path.join(params["input_path"], file), params
                        )
            else:
                self.process_single_file(params["input_path"], params)
        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}", exc_info=True)

    def process_single_file(self, file_path, params):
        try:
            self.logger.info(f"Starting processing of {file_path}")
            if not os.path.exists(file_path):
                raise AudioFileError(f"File not found: {file_path}")

            extension = os.path.splitext(file_path)[1].lower()
            if extension not in AUDIO_FILE_EXTENSIONS:
                raise AudioFileError(f"Unsupported file format: {extension}")

            # Create transcriber instance
            transcriber = AudioTranscriber(
                audio_file=file_path,
                output_dir=params["output_dir"],
                hf_token=params["hf_token"],
                skip_diarization=params["skip_diarization"],
                whisper_model=params["whisper_model"],
            )

            # Process the audio
            transcriber.process_audio(
                language=params["language"],
                min_speakers=params["min_speakers"],
                max_speakers=params["max_speakers"],
                use_vad=params["use_vad"],
                vad_method=params["vad_method"],
                verbose=params["verbose"],
                plot_word_alignment=params["plot_word_alignment"],
                detect_disfluencies=params["detect_disfluencies"],
                no_rename_speakers=params["no_rename_speakers"],
                num_processes=1,  # Always use 1 process for GUI operations
            )

            # Log success
            self.logger.info(f"Successfully processed: {os.path.basename(file_path)}")
            self.logger.info(
                f"Output saved in: {os.path.join(params['output_dir'], os.path.splitext(os.path.basename(file_path))[0])}"
            )

        except TranscriptionError as e:
            self.logger.error(f"Transcription error for {file_path}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(
                f"Unexpected error for {file_path}: {str(e)}", exc_info=True
            )
            raise

    def consume_logs(self):
        """Consume logs from the queue and update GUI"""
        while self.running:
            try:
                record = self.log_queue.get(timeout=0.1)
                if record:
                    msg = self.queue_handler.format(record)
                    self.after(0, self._update_log_display, msg)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing log: {e}")

    def _update_log_display(self, message):
        """Update log widget from main thread"""
        try:
            self.log_area.configure(state="normal")
            self.log_area.insert(tk.END, message + "\n")
            self.log_area.see(tk.END)
            self.log_area.configure(state="disabled")
        except tk.TclError:
            pass  # Widget might be destroyed

    def log(self, message):
        """Thread-safe logging"""
        self.logger.info(message)

    def on_closing(self):
        """Clean up when closing the application"""
        self.running = False
        if hasattr(self, "log_thread"):
            self.log_thread.join(timeout=1.0)
        self.destroy()


if __name__ == "__main__":
    app = Application()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
