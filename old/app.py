from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import sys
import tqdm
import urllib.request
import whisper
import whisper.transcribe

app = Flask(__name__, template_folder="template")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if request.method == "POST":
        f = request.files["file"]
        filename = secure_filename(f.filename)
        f.save(os.path.join("temp", filename))
        return "file uploaded successfully"


@app.route("/transcribe", methods=["POST"])
def transcribe_file():
    if request.method == "POST":
        filename = request.form.get("filename")
        file_path = os.path.join("temp", filename)
        result = translate(file_path)
        return result


class _CustomProgressBar(tqdm.tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current = self.n  # Imposta il valore iniziale

    def update(self, n):
        super().update(n)
        self._current += n  # Gestisci qui il progresso
        print("Progresso: " + str(self._current) + "/" + str(self.total))


# Inietta in tqdm.tqdm di Whisper, così possiamo vedere il progresso

transcribe_module = sys.modules["whisper.transcribe"]
transcribe_module.tqdm.tqdm = _CustomProgressBar

model = whisper.load_model("medium", device="cpu")

from pydub import AudioSegment


# convert to .wav
def convert_to_wav(audio_path):
    # Controlla l'estensione del file
    filename, file_extension = os.path.splitext(audio_path)
    if file_extension != ".wav":
        # Se il file non è in formato .wav, convertilo
        audio = AudioSegment.from_file(audio_path)
        new_audio_path = filename + ".wav"
        audio.export(new_audio_path, format="wav")
        print(f"File convertito in .wav e salvato come {new_audio_path}")
        return new_audio_path
    else:
        # Se il file è già in formato .wav, restituisci semplicemente il percorso del file originale
        print("Il file è già in formato .wav")
        return audio_path


def translate(audio_file, translate_language: str = None):
    options = dict(beam_size=5, best_of=5)
    translate_options = dict(task="translate", **options)
    result = model.transcribe(convert_to_wav(audio_file), **translate_options)
    if translate_language != None:
        result = whisper.translate(result["text"], target_language=translate_language)
    else:
        result = result["text"]
    return result


if __name__ == "__main__":
    if not os.path.exists("Transcriptor/temp"):
        # Se non esiste, creala
        print("creo cartella temp")
        os.makedirs("Transcriptor/temp")
    app.run(debug=True)
