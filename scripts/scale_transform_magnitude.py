import librosa
import numpy as np

import warnings

def compute_stm(
    y: np.ndarray,
    sr: int,
    target_sr: int = 8000,
    mel_flag: bool = True,
    log_flag: bool = True,
    detrend: bool = True,
    win_size: int = 256,
    hop: int = 160,
    n_mels: int = 40,
    with_padding: bool = True,
    oss_aggr=np.mean,
    autocor_window_type: str = "hamming",
    autocor_window_seconds: float = 8,
    autocor_hop_seconds: float = 0.5,
    autocor_norm_type: str = "max",
    autocor_norm_sum: bool = True,
):
    # validate parameters
    if autocor_window_seconds > (len(y) / sr): # REVIEW: what to do in this case?
        warnings.warn("auto_cor_window_seconds is bigger than duration of audio file, setting it to duration")
        autocor_window_seconds = len(y) / sr

    if y is None:
        raise ValueError("y is not valid")
    
    y, _ = librosa.effects.trim(y) # removing leading and trailing silence

    if sr != target_sr: # TODO: this can be done better if passing flag "resample"
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    if mel_flag:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=win_size, hop_length=hop, power=1)
    else:
        S = librosa.stft(y=y, n_fft=win_size, hop_length=hop)

    if log_flag:
        S = librosa.power_to_db(np.abs(S) ** 2, ref=np.max)

    oss = librosa.onset.onset_strength(
        S=S, sr=sr, detrend=detrend, aggregate=oss_aggr
    )  # computing onset strength signal

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
        librosa.fmt(oss_autocorrelation, beta=0.5, axis=0)
    )  # fast mellin transform

    return np.mean(scale_transform_magnitude, axis=1)  # return the mean over frames


def short_time_autocorrelation(
    y: np.array,
    win_size: int,
    hop_size: int,
    window_type: str,
    norm_sum: bool = True,
    with_padding: bool = True,
    norm_type: str = "max",
):
    if with_padding:
        pad_lenght = win_size // 2
        y = np.concatenate((np.zeros(pad_lenght), y, np.zeros(pad_lenght)))

    M = int(np.floor(len(y) - win_size) / hop_size)  # number of times window fits into signal

    window = get_window(window_type, win_size)  # get the window type
    A = np.zeros((win_size, M))  # initialize the autocorrelation matrix o be filled

    if norm_sum:
        lag_summands_num = np.arange(win_size, 0, -1)

    for i in range(M):
        # TODO: correlation can be re-written better
        # compute indices to select segment to self correlate
        start_idx = i * hop_size
        end_idx = start_idx + win_size
        segment = y[start_idx:end_idx] * window  # apply window to local segment
    
        segment_correlation = np.correlate(segment, segment, mode="full")[
            win_size - 1 :
        ]  # correlate local segment with itself and select positive lags

        # apply normalization
        if norm_sum:
            segment_correlation = segment_correlation / lag_summands_num

        if norm_type == "max":
            segment_correlation = segment_correlation / segment_correlation[0]

        elif norm_type == "min-max":
            segment_correlation = (segment_correlation - segment_correlation.min()) / (
                segment_correlation.max() - segment_correlation.min()
            )

        # Computing periodicity spectra as well
        # if fourier:
        #     S, _, _ = compute_tempogram_fourier(o_n, fs, N, H, Theta=theta, window=window, valid=valid)
        #     S = np.abs(S)
        #     #if valid:
        #     #    S = S[:,p1:-p2]
        #     for n in range(S.shape[1]):
        #         S[:, n] /= np.max(S[:, n])
        #     return np.mean(R, axis=1), np.mean(r, axis=1), np.mean(S, axis=1), R, r, S

        A[:, i] = segment_correlation # fill autocorrelation matrix

    return A


def zero_pad(y: np.array, window_size: int):
    pad_lenght = window_size // 2
    return np.pad(y, pad_lenght, mode="constant")


def get_window(window_type: str, window_size: int):
    if window_type == "hamming":
        return librosa.filters.get_window(window_type, window_size)
    elif window_type == "rectangular":
        return np.ones(window_size)
    else:
        raise ValueError("window_type should be one of 'hamming', 'rectangular'")
