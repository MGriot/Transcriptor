# %%
from pydub import AudioSegment
import whisper
import os
import tempfile
import shutil
from tqdm import tqdm
import natsort


# print(tempfile.gettempdir()) dove viene creata la cartella temporanea
class Transcriptor:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.audio_path = None
        self.audio = None
        self.temp_folder = None
        self.audio_chunk_paths = None

    def load_audio(self, path: str):
        self.audio_path = path
        self.audio = AudioSegment.from_file(path)
        self.audio_name = os.path.splitext(os.path.basename(path))[0]

    def split_audio_into_chunks(self):
        self.temp_folder = tempfile.mkdtemp()
        min_chunk_length = 30 * 1000  # 30 seconds
        start = 0
        end = min_chunk_length
        for _ in tqdm(range(0, len(self.audio), min_chunk_length)):
            end = min(end, len(self.audio))
            chunk = self.audio[start:end]
            chunk.export(f"{self.temp_folder}/chunk{start}.mp3", format="mp3")
            start += min_chunk_length
            end += min_chunk_length

        audio_chunk_paths = [
            os.path.join(self.temp_folder, f) for f in os.listdir(self.temp_folder)
        ]
        self.audio_chunk_paths = natsort.natsorted(audio_chunk_paths)

    def load_model(whisper_model: str, device: str = None):
        print("Initializing the Whisper model...")
        return whisper.load_model(whisper_model, device=device)

    def transcribe_audio(self, model: whisper):
        transcriptions = []
        error_count = 0
        for i, audio_chunk_path in enumerate(tqdm(self.audio_chunk_paths), start=1):
            audio_chunk = whisper.load_audio(audio_chunk_path)
            try:
                mel = whisper.log_mel_spectrogram(audio_chunk).to(model.device)
                try:
                    _, probs = model.detect_language(mel)
                    print(mel)
                    options = whisper.DecodingOptions(fp16=False)
                    result = whisper.decode(model, mel, options)
                    transcriptions.append(result.text)
                except:
                    result = model.transcribe(audio_chunk, fp16=False)
                    transcriptions.append(result["text"])
            except Exception:
                error_count += 1
                transcriptions.append(
                    f"\n Error in decoding audio from time {(i-1)*30/60}min <t< {i*30/60}min\n"
                )
        return transcriptions, error_count


# TODO: crea una classe che sfrutti queste funzioni


def obtain_file_name(audio_path):
    return os.path.splitext(os.path.basename(audio_path))[0]


def split_audio_into_chunks(input_file_path: str):
    """
    Split an audio file into chunks of at least 30 seconds in duration.

    This function creates a temporary folder to store the audio chunks, loads the specified audio file,
    and splits it into chunks of at least 30 seconds in duration. It returns the path to the temporary folder.

    :param input_file_path: Path to the audio file to split into chunks.
    :return: Path to the temporary folder containing the audio chunks.
    """
    # Create a temporary folder for audio chunks
    print("Creating a temporary folder")
    temp_folder = tempfile.mkdtemp()
    print(f"Temporary folder name:{temp_folder}")

    # Load the audio file
    print("Loading the audio file")
    audio = AudioSegment.from_file(input_file_path)

    # Split the audio file into chunks of at least 30 seconds in duration
    print("Splitting the audio file into 30-second chunks...")
    min_chunk_length = 30 * 1000  # 30 seconds
    start = 0
    end = min_chunk_length
    for _ in tqdm(range(0, len(audio), min_chunk_length)):
        end = min(
            end, len(audio)
        )  # se la condizione end<len audio è vera allora end = len(audio)
        chunk = audio[start:end]
        chunk.export(f"{temp_folder}/chunk{start}.mp3", format="mp3")
        start += min_chunk_length
        end += min_chunk_length
    return temp_folder


def initializing_model(whisper_model: str, device: str = None):
    """
    Initialize a Whisper model by loading it onto the specified device.

    :param whisper_model: The Whisper model to initialize. Can be "base" or "medium".
    :param device: The device to load the model onto. Can be "cpu" or "cuda".
    :return: Initialized Whisper model.
    """
    print("Initializing the Whisper model...")
    return whisper.load_model(whisper_model, device=device)


def get_audio_chunk_paths(audio_path: str):
    """
    Get a list of paths to audio chunks generated by splitting the specified audio file.

    :param audio_path: Path to the audio file to split into chunks.
    :return: List of paths to audio chunks.
    """
    temp_folder = split_audio_into_chunks(audio_path)
    audio_chunk_paths = [os.path.join(temp_folder, f) for f in os.listdir(temp_folder)]
    audio_chunk_paths = natsort.natsorted(audio_chunk_paths)
    return audio_chunk_paths, temp_folder


def transcribe_audio_chunks(audio_chunk_paths: str, model: whisper):
    """
    Transcribe a list of audio chunks using the specified model.

    :param audio_chunk_paths: List of paths to audio chunks to transcribe.
    :type audio_chunk_paths: list
    :param model: Model to use for transcription.
    :return: Tuple containing the list of transcriptions and the number of errors that occurred during transcription.
    """
    transcriptions = []
    error_count = 0
    for i, audio_chunk_path in enumerate(tqdm(audio_chunk_paths), start=1):
        audio_chunk = whisper.load_audio(audio_chunk_path)
        try:
            mel = whisper.log_mel_spectrogram(audio_chunk).to(model.device)
            try:
                _, probs = model.detect_language(mel)
                print(mel)
                options = whisper.DecodingOptions(fp16=False)
                result = whisper.decode(model, mel, options)
                transcriptions.append(result.text)
            except:
                result = model.transcribe(audio_chunk, fp16=False)
                transcriptions.append(result["text"])
        except Exception:
            error_count += 1
            transcriptions.append(
                f"\n Error in decoding audio from time {(i-1)*30/60}min <t< {i*30/60}min\n"
            )
    return transcriptions, error_count


def combine_text(transcriptions: str):
    return "".join(transcriptions)


def save_text(text: str, fname: str, dir_output: str = None, extension: str = "txt"):
    """
    Save a given text to a file with a specified file name, directory path, and file extension.

    :param text: The text to be saved to the file.
    :param fname: The name of the file to save the text to.
    :param dir_output: The path of the directory where the file will be saved. If not specified,
        the file will be saved in the current working directory.
    :param extension: The file extension to use when saving the file. Defaults to "txt".
    """
    if dir_output:
        file_path = os.path.join(dir_output, f"transcript_{fname}.{extension}")
    else:
        file_path = f"transcript_{fname}.{extension}"
    with open(file_path, "w") as f:
        f.write(text)


def transcribe_audio(audio_path: str, whisper_model: str, device: str = None):
    """
    Transcribe an audio file using the specified whisper model and device.

    :param audio_path: Path to the audio file to transcribe.
    :param whisper_model: Name of the whisper model to use for transcription, can be "base" or "medium", for more models see Whisper repository. Defaults to "base".
    :param device: Device to use for transcription, can be "cpu" or "cuda". device = "cuda" if torch.cuda.is_available() else "cpu"
    :return: Tuple containing the transcription and the file name of the audio file.
    """
    file_name = obtain_file_name(audio_path)
    print(f"File name: {file_name}")

    audio_chunk_paths, temp_folder = get_audio_chunk_paths(audio_path)
    model = initializing_model(whisper_model, device)

    print("Start transcribing the audio...")

    transcriptions, error_count = transcribe_audio_chunks(audio_chunk_paths, model)

    print("Combining the transcription into a single string")
    text = combine_text(transcriptions)

    print("Deleting the temporary folder")
    shutil.rmtree(temp_folder)

    print(f"Number of errors: {error_count}")

    return text, file_name


def transcriptor(
    audio_path: str,
    whisper_model: str,
    device: str = None,
    save_transcription: bool = True,
    dir_output: str = None,
    extension: str = "txt",
):
    text, file_name = transcribe_audio(audio_path, whisper_model, device)
    if save_transcription:
        save_text(
            text,
            file_name,
            dir_output,
            extension,
        )
    return text


# This code block is the main entry point of the script. It prompts the user to input the path of the
# audio file that needs to be transcribed and the pre-trained model to use for transcription (either
# "base" or "medium"). It then calls the `transcribe_audio` function with the provided input
# parameters and saves the transcribed text to a file using the `save_text` function.
# if __name__ == '__main__':
#    file_path=input("File path: ")
#    text,file_name=transcribe_audio(file_path, model=input("Model (base, medium): "), device=input("Device (cpu, cuda): "))
#    save_text(text=text,file_name=file_name)

# %%
