# Transcriptor
## English
It is a Python script that uses Whisper AI to generate the transcription of the audio file that is given as input. It can be used both from the terminal and as a Python module.

It is based on the Whisper library which you can find at the link: https://github.com/openai/whisper

In particular, the code uses the ‘pydub’ library to split an audio file into 30-second fragments and the ‘whisper’ library to transcribe each fragment. The code defines three main functions: ‘split_audio_file’, ‘transcribe_audio’ and ‘save_text’. The ‘split_audio_file’ function splits an audio file into 30-second fragments and saves them in a temporary folder. The ‘transcribe_audio’ function loads the audio fragments from the temporary folder, transcribes them using the Whisper model and combines the transcriptions into a single string. Finally, the ‘save_text’ function saves the transcription to a text file.


## Italian
È uno script in Python che sfrutta Whiper AI per generare la trascrizione del file audio che si da in ingresso. Può essere usato sia da terminale che come modulo Python.

Si basa sulla libreria Whisper che potete trovare al link: https://github.com/openai/whisper

In particolare il codice utilizza la libreria 'pydub' per dividere un file audio in frammenti di 30 secondi e la libreria 'whisper' per trascrivere ogni frammento. Il codice definisce tre funzioni principali: 'split_audio_file', 'transcribe_audio' e 'save_text'. La funzione 'split_audio_file' divide un file audio in frammenti di 30 secondi e li salva in una cartella temporanea. La funzione 'transcribe_audio' carica i frammenti audio dalla cartella temporanea, li trascrive utilizzando il modello *Whisper* e combina le trascrizioni in una singola stringa. Infine, la funzione 'save_text' salva la trascrizione in un file di testo.
