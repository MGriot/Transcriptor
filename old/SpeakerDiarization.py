# https://medium.com/analytics-vidhya/speaker-diarisation-89c963fa4fe8
import os
import pickle
import warnings
import librosa
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.mixture import *
import matplotlib.pyplot as plt


class SpeakerDiarization:
    def __init__(self):
        self.segLen, self.frameRate, self.numMix = 3, 50, 128
        self.speakers = []
        self.wavFile = None
        self.wavData = None

    def load_audio(self, path: str):
        self.wavFile = path
        self.wavData, _ = librosa.load(self.wavFile, sr=16000)
        vad = self.VoiceActivityDetection(self)
        mfcc = librosa.feature.mfcc(
            y=self.wavData, sr=16000, n_mfcc=20, hop_length=int(16000 / self.frameRate)
        ).T
        vad = np.reshape(vad, (len(vad),))
        if mfcc.shape[0] > vad.shape[0]:
            vad = np.hstack(
                (vad, np.zeros(mfcc.shape[0] - vad.shape[0]).astype("bool"))
            ).astype("bool")
        elif mfcc.shape[0] < vad.shape[0]:
            vad = vad[: mfcc.shape[0]]
        mfcc = mfcc[vad, :]

    def VoiceActivityDetection(self):
        # uses the librosa library to compute short-term energy
        ste = librosa.feature.rms(
            y=self.wavData, hop_length=int(16000 / self.frameRate)
        ).T
        thresh = 0.1 * (
            np.percentile(ste, 97.5) + 9 * np.percentile(ste, 2.5)
        )  # Trim 5% off and set threshold as 0.1x of the ste range
        return (ste > thresh).astype("bool")
