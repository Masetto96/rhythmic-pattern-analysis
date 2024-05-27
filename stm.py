import librosa
import numpy as np

# import essentia.standard as es
from scipy.signal import windows


def compute_stm(
    y: np.ndarray,
    sr: int,
    target_sr: int = 8000,
    mel_flag: bool = False,
    log_flag: bool = True,
    detrend: bool = True,
    win_size: int = 256,
    hop: int = 160,
    n_mels: int = 40,
    with_padding: bool = False,
    oss_aggr=np.median,
    autocor_window_type: str = "rectangular",
    auto_cor_window_seconds: float = 8,
    auto_cor_hop_seconds: float = 0.5,
    auto_cor_norm_type: str = "max",
    auto_cor_norm_sum: bool = True,
):
    """
    Computes the Scale Transform Magnitude (STM).

    Args:
        y (np.ndarray): Input signal
        sr (scalar): Original sampling rate
        target_sr (scalar): Target sampling rate (resampling)
        mel_flag (bool): If True, uses mel-spectrogram instead of short-time Fourier Transform (Default value = False)
        log_flag (bool): If True, log compresses the magnitude of the spectrogram (Default value = True)
        detrend (bool): Makes the spectral flux locally zero-meaned (Default value = True)
        win_size (int): Number of FFT points
        hop (int): Hop size
        n_mels (int): Number of channels in mel-scale mapping
        with_padding (bool): If True, zero-pads the onset strength signal before computing the autocorrelation (Default value = False)
        oss_aggr (callable): Aggregate onset strength signal
        autocor_window_type (str): Window type for the autocorrelation (Default value = "rectangular")
        auto_cor_window_seconds (float): Window size (seconds) for the autocorrelation (Default value = 8)
        auto_cor_hop_seconds (float): Hop size (seconds) for the autocorrelation (Default value = 0.5)
        auto_cor_norm_type (str): Normalization type for the autocorrelation (Default value = "max")
        auto_cor_norm_sum (bool): Normalizes by the number of summands in local autocorrelation (Default value = True)

    Returns:
        np.ndarray: Mean STM over frames
    """
    if auto_cor_window_seconds > (len(y) / sr):
        raise ValueError("auto_cor_window_seconds cannot be bigger than duration of audio file")

    if y is None:
        raise ValueError("y is not valid")

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # REVIEW: is this needed?
    y, _ = librosa.effects.trim(y)

    if mel_flag:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=win_size, hop_length=hop, power=1)
    else:
        S = librosa.stft(y=y, n_fft=win_size, hop_length=hop)

    if log_flag:
        S = librosa.power_to_db(np.abs(S) ** 2, ref=np.max)

    # REVIEW: what is essentia equivalent of this?
    # https://essentia.upf.edu/reference/std_SuperFluxNovelty.html?
    # https://essentia.upf.edu/reference/std_NoveltyCurve.html
    oss = librosa.onset.onset_strength(
        S=S, sr=sr, detrend=detrend, aggregate=oss_aggr
    )  # computing onset strength signal

    fs = sr / hop  # new sampling rate

    N = int(np.ceil(auto_cor_window_seconds * fs))  # window size for autocorrelation
    H = int(np.ceil(auto_cor_hop_seconds * fs))  # hop size (lag) for autocorrelation

    if with_padding:  # zero pad oss before autocorrelation to center the window at the start of the signal
        oss = zero_pad(y=oss, window_size=N)

    oss_autocorrelation = short_time_autocorrelation(
        y=oss,
        win_size=N,
        hop_size=H,
        window_type=autocor_window_type,
        norm_sum=auto_cor_norm_sum,
        norm_type=auto_cor_norm_type,
    )

    scale_transform_magnitude = np.abs(
        librosa.fmt(oss_autocorrelation, beta=0.5, axis=0)
    )  # fast mellin transform

    return np.mean(scale_transform_magnitude, axis=1)  # return the mean over frames


def short_time_autocorrelation(
    y: np.array, win_size: int, hop_size: int, window_type: str, norm_sum: bool = True, norm_type: str = "max"
):
    """
    Compute the short-time autocorrelation matrix of a signal given a window and a hop size.
    Finally, applies normalization.

    Args:
        y (np.array): Input signal
        win_size (int): Size of the window used for autocorrelation
        hop_size (int): Hop size used for autocorrelation
        window_type (str): Type of window used for autocorrelation (e.g. "hamming", "rectangular")
        norm_sum (bool): If True, normalizes the autocorrelation matrix by the number of summands
        norm_type (str): Type of normalization used (e.g. "max", "min-max")

    Returns:
        A (np.ndarray): normalized autocorrelation matrix
    """
    remanining_lenght = len(y) - win_size
    M = remanining_lenght // hop_size + 1  # number of times window fits the signal

    window = get_window(window_type, win_size)  # get the window type
    A = np.zeros((win_size, M))  # initialize the autocorrelation matrix

    if norm_sum:
        lag_summands_num = np.arange(win_size, 0, -1)

    for i in range(M):
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

        # fill autocorrelation matrix
        A[:, i] = segment_correlation

    return A


def zero_pad(y: np.array, window_size: int):
    pad_lenght = window_size // 2
    return np.pad(y, pad_lenght, mode="constant")


def get_window(window_type: str, window_size: int):
    if window_type == "hamming":
        return windows.hamming(window_size)
    elif window_type == "rectangular":
        return np.ones(window_size)
    else:
        raise ValueError("window_type should be one of 'hamming', 'rectangular'")
