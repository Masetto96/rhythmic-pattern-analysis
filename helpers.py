import librosa
import numpy as np

def compute_stm(y, sr, win_size=1024, hop_size=512, n_mels=128):
    """
    Compute the scale transform magnitude (STM) for the input audio signal.

    Args:
        y (numpy.ndarray): The input audio signal.
        sr (int): The sample rate of the audio signal.
        win_size (int, optional): The window size for the Short-Time Fourier Transform (STFT). 
            Defaults to 1024.
        hop_size (int, optional): The hop size for the STFT. Defaults to 512.
        n_mels (int, optional): The number of Mel filterbanks. Defaults to 128.

    Returns:
        numpy.ndarray: The STM.
    """
    # Fourier
    spectrogram = librosa.stft(y, win_length=win_size, hop_length=hop_size)

    # Decompose into harmonic and percussive spectrogram
    _, percussive_spectro = librosa.decompose.hpss(S=spectrogram)

    # Percussive mel spectrogram
    percussive_mel_spec = librosa.feature.melspectrogram(S=percussive_spectro, n_mels=n_mels)
    percussive_mel_spec = librosa.amplitude_to_db(np.abs(percussive_mel_spec), ref=np.max)

    # Onset strength
    oss = librosa.onset.onset_strength(S=percussive_mel_spec, aggregate=np.median, detrend=True, lag=3)

    # 8 seconds max lag for autocorrelation
    oss_ac = librosa.autocorrelate(oss, max_size= 8 * sr // hop_size)

    # Normalize the autocorrelation
    oss_ac = librosa.util.normalize(oss_ac, norm=np.inf)

    # Scale transform magnitude
    return librosa.fmt(oss_ac)


