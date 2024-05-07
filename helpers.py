import librosa
import numpy as np

def compute_stm(y:np.array, sr, win_size=256, hop=128, n_mels=50, auto_cor_lag_seconds:int = 8):

    if sr != 8000:
        y = librosa.resample(y, orig_sr=sr, target_sr=8000)
        sr = 8000
    
    y, _ = librosa.effects.trim(y)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, win_length=win_size, hop_length=hop)

    oss = librosa.onset.onset_strength(S=librosa.power_to_db(mel, ref=np.max), sr=sr, lag=2, aggregate=np.median)

    oss_ac = librosa.autocorrelate(oss, max_size= auto_cor_lag_seconds * sr // hop)

    oss_ac = librosa.util.normalize(oss_ac, norm=np.inf)

    stm = np.abs(librosa.fmt(oss_ac))

    return stm

