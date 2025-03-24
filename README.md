# Audio Transcriptor

A Python-based audio transcription tool that combines speaker diarization and speech-to-text capabilities. This tool can process audio files, identify different speakers, and generate transcriptions with speaker labels and timestamps.

## Features

- Speaker diarization using pyannote.audio
- Audio chunking based on speaker segments
- Transcription with timestamps for each word
- GUI interface for easy operation
- Support for multiple audio formats
- Export transcriptions in JSON format

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for better performance)
- FFmpeg installed and available in system PATH
- Hugging Face account and API token

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Transcriptor
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Create a `config.py` file in the root directory with your Hugging Face API token:
```python
HF_TOKEN = 'your_hugging_face_token_here'
```

## Configuration

1. Get your Hugging Face API token:
   - Visit [Hugging Face](https://huggingface.co/)
   - Create an account or sign in
   - Go to Settings > Access Tokens
   - Create a new token

2. Accept the license terms for the following models:
   - [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
   - [faster-whisper](https://huggingface.co/guillaumekln/faster-whisper-large-v2)

## Usage

### GUI Interface

Run the GUI application:
```bash
python audio_transcriber_gui.py
```

### Command Line Interface

Run the transcriber from command line:
```bash
python audio_transcriber.py path/to/your/audio/file
```

## Output

The tool generates:
- Chunked audio files for each speaker segment
- JSON files containing:
  - Speaker identification
  - Timestamps
  - Transcribed text
  - Word-level timing information

## Project Structure

```
Transcriptor/
├── audio_transcriber.py      # Core transcription logic
├── audio_transcriber_gui.py  # GUI interface
├── config.py                 # Configuration file
├── requirements.txt          # Python dependencies
└── chunks/                   # Output directory for processed files
```

## License

[Your chosen license]

## Contributing

[Your contribution guidelines]

## English
It is a Python script that uses Whisper AI to generate the transcription of the audio file that is given as input. It can be used both from the terminal and as a Python module.

It is based on the **Whisper** [library](https://github.com/openai/whisper) and takes inspiration from the [video](https://www.youtube.com/watch?v=E0_kG5j6lEo&t=1121s&ab_channel=G-Program-It) by **G-Program-It**.

In particular, the code uses the `pydub` library to split an audio file into 30-second fragments and the `whisper` library to transcribe each fragment. The code defines three main functions: `split_audio_file`, `transcribe_audio` and `save_text`. The `split_audio_file` function splits an audio file into 30-second fragments and saves them in a temporary folder. The `transcribe_audio` function loads the audio fragments from the temporary folder, transcribes them using the Whisper model and combines the transcriptions into a single string. Finally, the `save_text` function saves the transcription to a text file.
### Setup
### How to use
#### Example
### Code description
Here`s a step-by-step explanation of the code:

1. The code starts by importing the necessary libraries: `pydub`, `whisper`, `os`, `tempfile`, `shutil`, `tqdm`, and `natsort`.
2. The `split_audio_file` function is defined. This function takes an input file path as an argument and splits the audio file into 30-second fragments.
   1. A temporary folder is created to store the audio fragments.
   2. The audio file is loaded using the `AudioSegment.from_file` method from the `pydub` library.
   3. The audio file is split into 30-second fragments using a for loop. Each fragment is exported as an mp3 file and saved in the temporary folder.
   4. The function returns the path to the temporary folder.
3. The whisper model is initialized using the `whisper.load_model method`.
4. The `transcribe_audio` function is defined. This function takes an audio file path as an argument and transcribes the audio using the `whisper` library.
5. The audio file is split into 30-second fragments using the previously defined `split_audio_file` function.
   1. The audio fragment paths are sorted in ascending order using the `natsort.natsorted` method.
   2. A for loop is used to iterate over each audio fragment path. For each fragment:
      1. The audio fragment is loaded using the whisper.load_audio method and a log-Mel spectrogram is created using the whisper.log_mel_spectrogram method.
      2. The language of the audio fragment is detected using the model.detect_language method.
      3. The audio fragment is decoded (i.e., transcribed) using the whisper.decode method and the transcription is appended to a list of transcriptions.
      4. If an error occurs during decoding, an error message is appended to the list of transcriptions instead.
   3. The transcriptions are combined into a single string using the .join() method on the list of transcriptions.
   4. The temporary folder and its files are deleted using the shutil.rmtree method.
   5. The function returns the transcription text and file name.
6. The save_text function is defined. This function takes text, a file name, an optional directory path, and an optional file extension as arguments and saves the text to a file with the specified name, directory, and extension.


## Italiano
È uno script in Python che sfrutta Whiper AI per generare la trascrizione del file audio che si da in ingresso. Può essere usato sia da terminale che come modulo Python.

Si basa sulla libreria Whisper che potete trovare al link: https://github.com/openai/whisper e prende ispirazione dal video di G-Program-It: https://www.youtube.com/watch?v=E0_kG5j6lEo&t=1121s&ab_channel=G-Program-It.

In particolare il codice utilizza la libreria 'pydub' per dividere un file audio in frammenti di 30 secondi e la libreria 'whisper' per trascrivere ogni frammento. Il codice definisce tre funzioni principali: 'split_audio_file', 'transcribe_audio' e 'save_text'. La funzione 'split_audio_file' divide un file audio in frammenti di 30 secondi e li salva in una cartella temporanea. La funzione 'transcribe_audio' carica i frammenti audio dalla cartella temporanea, li trascrive utilizzando il modello *Whisper* e combina le trascrizioni in una singola stringa. Infine, la funzione 'save_text' salva la trascrizione in un file di testo.

### Installazione

#### Metodo rapido

##### Anaconda

Se si utilizza Anaconda, è possibile creare l`ambiente virtuale con il seguente comando:

```shell
conda env create -f environment.yml
```

##### Pip

Se si preferisce utilizzare pip per creare l`ambiente virtuale, seguire questi passaggi:

1. Crea un nuovo ambiente virtuale con il comando:

   ```shell
   python -m venv name_env
   ```

2. Attiva l`ambiente virtuale:
   - Su Windows:

      ```shell
      name_env\Scripts\activate
      ```

   - Su macOS/Linux:

      ```shell
      source name_env/bin/activate
      ```

3. Installa le dipendenze con il comando:

   ```shell
   pip install -r requirements.txt
   ```

#### Metodo dettagliato

##### Anaconda

Per creare un ambiente virtuale utilizzando Anaconda, seguire questi passaggi:

1. Crea un nuovo ambiente virtuale senza pacchetti predefiniti utilizzando il comando:

   ```shell
   conda create --no-default-packages -n transcriptor python=3.10
   ```

2. Attiva l`ambiente virtuale con il comando:

   ```
   conda activate transcriptor
   ```

3. Installa le librerie necessarie utilizzando i seguenti comandi:

   ```shell
   conda install -c conda-forge pydub

   pip install -U openai-whisper

   conda install -c anaconda natsort
   ```

Se si desidera utilizzare il sistema CUDA per trascrivere il testo, è necessario installare CUDA dal [sito ufficiale](https://developer.nvidia.com/cuda-downloads) e installare pytorch-cuda utilizzando i seguenti comandi:

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda install torchaudio -c pytorch -c conda-forge
```

Per verificare che l`installazione sia andata a buon fine e che il sistema CUDA sia disponibile, è possibile utilizzare il seguente codice Python:

```python
# check if the CUDA system is available
import torch
torch.cuda.is_available()
# True if the CUDA system is available
```

*Parte in via di sviluppo*

Nel caso si voglia migliorare la qualità del file audio serve installare ancora le seguenti librerie:

```shell
pip install speechbrain #per la manipolazione dell'audio e migliorarlo

conda install -c conda-forge pysoundfile
```

Per la parte di Speakers Diarization è necessario installare:
```
scoop install libsndfile
conda install -c conda-forge librosa


pip install -U git+https://github.com/pyannote/pyannote-audio.git
```