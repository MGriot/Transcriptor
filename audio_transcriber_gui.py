# done with deepseeker R1 + search

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import queue
import sys
import threading
import os
import logging
from audio_transcriber import (
    process_single_audio,
    DEFAULT_OUTPUT_DIR,
    AUDIO_FILE_EXTENSIONS,
    VAD_METHODS,
)


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Transcriber GUI")
        self.queue = queue.Queue()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.setup_gui()
        self.setup_logging_redirection()

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

        # Process Button
        ttk.Button(self, text="Start Processing", command=self.start_processing).grid(
            row=3, column=0, pady=10
        )

        # Log Output
        self.log_area = scrolledtext.ScrolledText(self, state="disabled", height=15)
        self.log_area.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")

        # Configure grid weights
        self.grid_rowconfigure(4, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def setup_logging_redirection(self):
        sys.stdout = self
        sys.stderr = self

    def write(self, text):
        self.queue.put(text)

    def flush(self):
        pass

    def browse_input(self):
        path = (
            filedialog.askopenfilename(title="Select Audio File")
            if tk.messagebox.askyesno("Input Type", "Select a file? (No for directory)")
            else filedialog.askdirectory(title="Select Audio Directory")
        )
        if path:
            self.input_path.set(path)

    def browse_output(self):
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_dir.set(path)

    def start_processing(self):
        params = {
            "input_path": self.input_path.get(),
            "output_dir": self.output_dir.get(),
            "hf_token": self.hf_token.get() or None,
            "skip_diarization": self.skip_diarization.get(),
            "whisper_model": self.whisper_model.get(),
            "language": self.language.get() or None,
            "min_speakers": (
                int(self.min_speakers.get()) if self.min_speakers.get() else None
            ),
            "max_speakers": (
                int(self.max_speakers.get()) if self.max_speakers.get() else None
            ),
            "use_vad": self.use_vad.get(),
            "vad_method": self.vad_method.get() if self.vad_method.get() else None,
            "verbose": self.verbose.get(),
            "plot_word_alignment": self.plot_word_alignment.get(),
            "detect_disfluencies": self.detect_disfluencies.get(),
            "no_rename_speakers": self.no_rename_speakers.get(),
        }

        if not os.path.exists(params["input_path"]):
            self.log("Error: Invalid input path")
            return

        processing_thread = threading.Thread(
            target=self.process_files, args=(params,), daemon=True
        )
        processing_thread.start()
        self.after(100, self.update_log)

    def process_files(self, params):
        try:
            if os.path.isdir(params["input_path"]):
                for file in os.listdir(params["input_path"]):
                    if file.lower().endswith(AUDIO_FILE_EXTENSIONS):
                        self.process_single_file(
                            os.path.join(params["input_path"], file), params
                        )
            else:
                self.process_single_file(params["input_path"], params)
        except Exception as e:
            self.log(f"Processing error: {str(e)}")

    def process_single_file(self, file_path, params):
        try:
            process_single_audio(
                audio_file_path=file_path,
                output_dir=params["output_dir"],
                hf_token=params["hf_token"],
                skip_diarization=params["skip_diarization"],
                whisper_model=params["whisper_model"],
                language=params["language"],
                min_speakers=params["min_speakers"],
                max_speakers=params["max_speakers"],
                use_vad=params["use_vad"],
                vad_method=params["vad_method"],
                verbose=params["verbose"],
                plot_word_alignment=params["plot_word_alignment"],
                detect_disfluencies=params["detect_disfluencies"],
                no_rename_speakers=params["no_rename_speakers"],
            )
            self.log(f"Processed: {os.path.basename(file_path)}")
        except Exception as e:
            self.log(f"Error processing {file_path}: {str(e)}")

    def update_log(self):
        while not self.queue.empty():
            msg = self.queue.get_nowait()
            self.log_area.configure(state="normal")
            self.log_area.insert(tk.END, msg)
            self.log_area.yview(tk.END)
            self.log_area.configure(state="disabled")
        self.after(100, self.update_log)

    def log(self, message):
        self.log_area.configure(state="normal")
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.configure(state="disabled")


if __name__ == "__main__":
    app = Application()
    app.mainloop()
