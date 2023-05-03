from pydub import AudioSegment
import whisper
import os
import tempfile
import shutil
from tqdm import tqdm
import natsort 

#print(tempfile.gettempdir()) dove viene creata la cartella temporanea

def split_audio_file(input_file_path):
    # Crea una cartella temporanea per i frammenti audio
    print("Creating a temporary folder")
    temp_folder = tempfile.mkdtemp()
    print(f"Temporary folder name:{temp_folder}")

    # Carica il file audio
    print("Loading the audio file")
    audio = AudioSegment.from_file(input_file_path)

    # Dividi il file audio in frammenti di durata almeno 30 secondi
    print("Splitting the audio file into 30-second fragments")
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

print("Initializing the Whisper model...")
model = whisper.load_model("base", device="cpu") #"medium"

def transcribe_audio(audio_path):
    file_name=os.path.splitext(os.path.basename(audio_path))[0]
    print(f"File name: {file_name}")
    # Dividi l'audio in frammenti di durata massima di 30 secondi
    temp_folder = split_audio_file(audio_path)
    audio_chunk_paths = [os.path.join(temp_folder, f) for f in os.listdir(temp_folder)]
    audio_chunk_paths=natsort.natsorted(audio_chunk_paths) # orino in maniera crescente i filename

    # Trascrivi i frammenti audio
    print("Start transcribing the audio")
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
    if directory_path:
        file_path = os.path.join(directory_path, f'transcript_{file_name}.{extension}')
    else:
        file_path = f'transcript_{file_name}.{extension}'
    with open(file_path, 'w') as f:
        f.write(text)
