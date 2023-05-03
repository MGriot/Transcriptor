from pydub import AudioSegment
import whisper
import os
import tempfile
import shutil
from tqdm import tqdm
import natsort 

#print(tempfile.gettempdir()) dove viene creata la cartella temporanea

def split_audio_file(input_file_path):
    """
    The function splits an audio file into 30-second fragments and saves them in a temporary folder.
    
    :param input_file_path: The path of the audio file that needs to be split into smaller chunks
    :return: the path of the temporary folder where the audio file has been split into 30-second
    fragments.
    """
    # Crea una cartella temporanea per i frammenti audio
    print("Creating a temporary folder")
    temp_folder = tempfile.mkdtemp()
    print(f"Temporary folder name:{temp_folder}")

    # Carica il file audio
    print("Loading the audio file")
    audio = AudioSegment.from_file(input_file_path)

    # Dividi il file audio in frammenti di durata almeno 30 secondi
    print("Splitting the audio file into 30-second fragments...")
    min_chunk_length = 30 * 1000 # 30 secondi
    start = 0
    end = min_chunk_length
    for _ in tqdm(range(0, len(audio), min_chunk_length)):
        if end > len(audio):
            end = len(audio)
        chunk = audio[start:end]
        chunk.export(f"{temp_folder}/chunk{start}.mp3", format="mp3")
        start += min_chunk_length
        end += min_chunk_length

    return temp_folder

def initializati_model(model):
    """
    This function initializes a Whisper model by loading it onto the CPU.
    
    :param model: The model parameter is the name or path of the pre-trained Whisper model that needs to
    be loaded
    """
    print("Initializing the Whisper model...")
    model = whisper.load_model(model, device="cpu") 

def transcribe_audio(audio_path, model="base"):
    """
    This function transcribes an audio file by splitting it into chunks, detecting the language,
    decoding the audio, and combining the transcriptions into a single string.
    
    :param audio_path: The path to the audio file that needs to be transcribed
    :param model: The model parameter specifies which pre-trained model to use for transcription. It can
    be either "base" or "medium", as per the Whisper documentation, defaults to base (optional)
    :return: a tuple containing the transcribed text and the file name of the audio file.
    """
    file_name=os.path.splitext(os.path.basename(audio_path))[0]
    print(f"File name: {file_name}")
    # Dividi l'audio in frammenti di durata massima di 30 secondi
    temp_folder = split_audio_file(audio_path)
    audio_chunk_paths = [os.path.join(temp_folder, f) for f in os.listdir(temp_folder)]
    audio_chunk_paths=natsort.natsorted(audio_chunk_paths) # orino in maniera crescente i filename
    initializati_model(model) #"base" o "medium" see Whisper documentation

    # Trascrivi i frammenti audio
    print("Start transcribing the audio...")
    transcriptions = []
    i=0
    error_count = 0
    for audio_chunk_path in tqdm(audio_chunk_paths):
        i+=1
        # Carica il frammento audio e crea uno spettrogramma log-Mel
        audio_chunk = whisper.load_audio(audio_chunk_path)
        try:
            mel = whisper.log_mel_spectrogram(audio_chunk).to(model.device)

            # rileva la lingua parlata
            _, probs = model.detect_language(mel)
            #print(f"Lingua rilevata: {max(probs, key=probs.get)}")

            # decodifica l'audio
            options = whisper.DecodingOptions(fp16=False)
            result = whisper.decode(model, mel, options)

            transcriptions.append(result.text)
        except Exception:
            error_count += 1
            transcriptions.append(f"\n Error in decoding audio from time {(i-1)*30/60}min <t< {i*30/60}min\n")
    # Combina le trascrizioni in una singola stringa
    print("Combining the transcription into a single string")
    text="".join(transcriptions)
    # Elimina la cartella temporanea e i suoi file
    print("Deleting the temporary folder")
    shutil.rmtree(temp_folder)
    
    print(f"Number of errors: {error_count}")
    return text,file_name

def save_text(text, file_name, directory_path=None, extension="txt"):
    """
    This function saves a given text to a file with a specified file name, directory path, and file
    extension.
    
    :param text: The text that you want to save to a file
    :param file_name: The name of the file to be saved
    :param directory_path: The directory path where the file will be saved. If this parameter is not
    provided, the file will be saved in the current working directory
    :param extension: The file extension to use for the saved file. By default, it is set to "txt",
    defaults to txt (optional)
    """
    if directory_path:
        file_path = os.path.join(directory_path, f'transcript_{file_name}.{extension}')
    else:
        file_path = f'transcript_{file_name}.{extension}'
    with open(file_path, 'w') as f:
        f.write(text)


# This code block is the main entry point of the script. It prompts the user to input the path of the
# audio file that needs to be transcribed and the pre-trained model to use for transcription (either
# "base" or "medium"). It then calls the `transcribe_audio` function with the provided input
# parameters and saves the transcribed text to a file using the `save_text` function.
if __name__ == '__main__':
    file_path=input("File path: ")
    text,file_name=transcribe_audio(file_path, model=input("Model (base, medium): "))
    save_text(text=text,file_name=file_name)
