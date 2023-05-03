# Transcriptor

è uno script in Python che sfrutta Whiper AI per generare la trascrizione del file audio che si da in ingresso. Può essere usato sia da terminale che come modulo Python.

Si basa sulla libreria Whisper che potete trovare al link: https://github.com/openai/whisper

In particolare il codice utilizza la libreria 'pydub' per dividere un file audio in frammenti di 30 secondi e la libreria 'whisper' per trascrivere ogni frammento. Il codice definisce tre funzioni principali: 'split_audio_file', 'transcribe_audio' e 'save_text'. La funzione 'split_audio_file' divide un file audio in frammenti di 30 secondi e li salva in una cartella temporanea. La funzione 'transcribe_audio' carica i frammenti audio dalla cartella temporanea, li trascrive utilizzando il modello *Whisper* e combina le trascrizioni in una singola stringa. Infine, la funzione 'save_text' salva la trascrizione in un file di testo.
