import warnings
import librosa

import numpy as np
from .helpers import (
    short_time_autocorrelation,
    compute_tempogram_fourier,
    generate_spectrogram,
)
from scipy.ndimage import gaussian_filter1d

def compute_stm(
    y: np.ndarray = None,
    sr: int = 44100,
    target_sr: int = 8000,
    mel_flag: bool = True,
    log_flag: bool = True,
    detrend: bool = True,
    win_size: int = 256,
    hop: int = 160,
    n_mels: int = 40,
    with_padding: bool = True,
    autocor_window_type: str = "hamming",
    autocor_window_seconds: float = 8,
    autocor_hop_seconds: float = 0.5,
    autocor_norm_type: str = "max",
    autocor_norm_sum: bool = False,
    num_stm_coefs: int = 200,
    with_tempogram=True,
    sigma=1,
):
    # TODO: is this needed?
    # y, _ = librosa.effects.trim(y)  # removing leading and trailing silence

    # validate parameters
    if autocor_window_seconds > (len(y) / sr):  # REVIEW: what to do in this case?
        warnings.warn(
            "auto_cor_window_seconds is bigger than duration of audio file, setting it to duration"
        )
        autocor_window_seconds = len(y) / sr

    if y is None:
        raise ValueError("y is not valid")

    if (
        sr != target_sr
    ):  # TODO: this can be done better if passing flag "with_resample" and audio_file_path
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # S = generate_spectrogram(y, sr, mel_flag, log_flag, win_size, hop, n_mels)
    # oss = librosa.onset.onset_strength(
    #     S=S, sr=sr, detrend=detrend
    # )  # computing onset strength signal

    oss = librosa.onset.onset_strength(
        y=y, sr=sr, n_fft=win_size, hop_length=hop, detrend=detrend, n_mels=n_mels
    )
    if sigma > 0:
        oss = gaussian_filter1d(oss, sigma)

    fs = sr / hop  # new sampling rate
    N = int(np.ceil(autocor_window_seconds * fs))  # window size for autocorrelation
    H = int(np.ceil(autocor_hop_seconds * fs))  # hop size (lag) for autocorrelation
    oss_autocorrelation = short_time_autocorrelation(
        y=oss,
        with_padding=with_padding,
        win_size=N,
        hop_size=H,
        window_type=autocor_window_type,
        norm_sum=autocor_norm_sum,
        norm_type=autocor_norm_type,
    )

    scale_transform_magnitude = np.abs(
        librosa.fmt(oss_autocorrelation, beta=0.5, axis=0, t_min=0.5, over_sample=1)
    )  # fast mellin transform

    # Computing periodicity spectra as well
    if with_tempogram:
        tempogram, _, _ = compute_tempogram_fourier(
            oss, fs, N, H, window=autocor_window_type, valid=with_padding
        )
        tempogram = np.abs(tempogram)

        for n in range(tempogram.shape[1]):
            tempogram[:, n] /= np.max(tempogram[:, n])

        return (
            np.mean(scale_transform_magnitude, axis=1)[:num_stm_coefs],
            np.mean(oss_autocorrelation, axis=1),
            scale_transform_magnitude,
            oss_autocorrelation,
            np.mean(tempogram, axis=1),
            tempogram,
        )

    return (
        np.mean(scale_transform_magnitude, axis=1)[:num_stm_coefs],
        np.mean(oss_autocorrelation, axis=1),
        scale_transform_magnitude,
        oss_autocorrelation,
        _,
        _,
    )  # return the mean over frames


def compute_stm_multi_channel(
    y: np.ndarray = None,
    sr: int = 44100,
    channels=[0, 5, 20, 40],
    target_sr: int = 8000,
    mel_flag: bool = True,
    log_flag: bool = True,
    detrend: bool = True,
    win_size: int = 256,
    hop: int = 160,
    n_mels: int = 40,
    with_padding: bool = True,
    autocor_window_type: str = "hamming",
    autocor_window_seconds: float = 8,
    autocor_hop_seconds: float = 0.5,
    autocor_norm_type: str = "max",
    autocor_norm_sum: bool = False,
    num_stm_coefs: int = 200,
    sigma=1,
):
    # TODO: is this needed?
    y, _ = librosa.effects.trim(y)  # removing leading and trailing silence

    # validate parameters
    if autocor_window_seconds > (len(y) / sr):  # REVIEW: what to do in this case?
        warnings.warn(
            "auto_cor_window_seconds is bigger than duration of audio file, setting it to duration"
        )
        autocor_window_seconds = len(y) / sr

    if y is None:
        raise ValueError("y is not valid")

    if sr != target_sr:  # TODO: this can be done better if passing flag "with_resample"
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # S = generate_spectrogram(y, sr, mel_flag, log_flag, win_size, hop, n_mels)

    oss_multi = librosa.onset.onset_strength_multi(
        y=y,
        sr=sr,
        detrend=detrend,
        channels=channels,
        hop_length=hop,
        n_fft=win_size,
        n_mels=n_mels,
    )  # computing onset strength signal on multiple channels

    if sigma > 0:
        oss_multi = gaussian_filter1d(oss_multi, sigma, axis=-1)

    fs = sr / hop  # new sampling rate
    N = int(np.ceil(autocor_window_seconds * fs))  # window size for autocorrelation
    H = int(np.ceil(autocor_hop_seconds * fs))  # hop size (lag) for autocorrelation

    R = []
    for oss_channel in oss_multi:
        oss_channel_autocorr = short_time_autocorrelation(
            y=oss_channel,
            with_padding=with_padding,
            win_size=N,
            hop_size=H,
            window_type=autocor_window_type,
            norm_sum=autocor_norm_sum,
            norm_type=autocor_norm_type,
        )
        stm = np.abs(librosa.fmt(oss_channel_autocorr, axis=0, beta=0.5))
        R.append(np.mean(stm, axis=1)[:num_stm_coefs])

    R = np.vstack(R)
    return np.mean(R, axis=0), R
