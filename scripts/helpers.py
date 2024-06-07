import numpy as np
import librosa
import pandas as pd
from bokeh.plotting import figure, output_file
from bokeh.models import ColumnDataSource, CustomJS, HoverTool
from bokeh.io import output_notebook
from bokeh.transform import linear_cmap
from bokeh.palettes import Category20_10


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

    M = (
        int(np.floor(len(y) - win_size) / hop_size) + 1
    )  # number of times window fits into signal

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

        A[:, i] = segment_correlation  # fill autocorrelation matrix

    return A


def get_window(window_type: str, window_size: int):
    if window_type == "hamming":
        return librosa.filters.get_window(window_type, window_size)
    elif window_type == "rectangular":
        return np.ones(window_size)
    else:
        raise ValueError("window_type should be one of 'hamming', 'rectangular'")


def compute_tempogram_fourier(
    x, Fs, N, H, Theta=np.arange(30, 601, 1), window="hann", fftbins=True, valid=False
):
    """Compute Fourier-based tempogram [FMP, Section 6.2.2]

    Notebook: C6/C6S2_TempogramFourier.ipynb

    Args:
        x (np.ndarray): Input signal
        Fs (scalar): Sampling rate
        N (int): Window length
        H (int): Hop size
        Theta (np.ndarray): Set of tempi (given in BPM) (Default value = np.arange(30, 601, 1))
        window (str): Name of the window function
        fftbins (bool): Whether window is periodic (True) or symmetric (False)
        valid (bool): Computes autocorrelation without padding (Default value = False)

    Returns:
        X (np.ndarray): Tempogram
        T_coef (np.ndarray): Time axis (seconds)
        F_coef_BPM (np.ndarray): Tempo axis (BPM)
    """
    # This code was adapted from Meinard Müller's FMP notebooks, which are licensed under a
    # MIT License: https://opensource.org/licenses/MIT
    # https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S2_TempogramFourier.html

    # win = np.hanning(N)
    win = librosa.filters.get_window(window, N, fftbins=fftbins)
    if valid == False:
        N_left = N // 2
        L = x.shape[0]
        L_left = N_left
        L_right = N_left
        L_pad = L + L_left + L_right
        # x_pad = np.pad(x, (L_left, L_right), 'constant')  # doesn't work with jit
        x_pad = np.concatenate((np.zeros(L_left), x, np.zeros(L_right)))
    else:
        L_pad = x.shape[0]
        x_pad = x
    t_pad = np.arange(L_pad)
    M = int(np.floor(L_pad - N) / H) + 1
    K = len(Theta)
    X = np.zeros((K, M), dtype=np.complex_)

    for k in range(K):
        omega = (Theta[k] / 60) / Fs
        exponential = np.exp(-2 * np.pi * 1j * omega * t_pad)
        x_exp = x_pad * exponential
        for n in range(M):
            t_0 = n * H
            t_1 = t_0 + N
            X[k, n] = np.sum(win * x_exp[t_0:t_1])
    if valid == False:
        T_coef = np.arange(M) * H / Fs
    else:
        T_coef = np.arange(M) * H / Fs + (N // 2) / Fs
    F_coef_BPM = Theta
    return X, T_coef, F_coef_BPM


def display_interactive_scatter_plot(
    embeddings2d,
    labels,
    hover_data: pd.DataFrame,
    output_filename: str = "scatter_interactive.html",
):
    """
    Displays an interactive scatter plot with UMAP embeddings and audio playback on click.

    Parameters:
    - embeddings2d: 2D array or DataFrame with UMAP embeddings (shape: [n_samples, 2])
    - hover_data: DataFrame containing additional information for hover tooltips and audio file paths
    - output_filename: Filename for saving the plot as an HTML file
    """

    # Prepare the data dictionary combining embeddings and hover data
    data = {
        "x": embeddings2d[:, 0],
        "y": embeddings2d[:, 1],
        "absolute_path": hover_data["absolute_path"],
        "genre": hover_data["genre"],
        "style": hover_data["style"],
        "audio_filename": hover_data["audio_filename"],
        "label": labels,
    }
    color_mapper = linear_cmap(
        field_name="label", palette=Category20_10, low=min(labels), high=max(labels)
    )
    # Create a ColumnDataSource from the dictionary
    source = ColumnDataSource(data=data)

    # Create a scatter plot with navigation tools
    tools = "tap, wheel_zoom, box_zoom, pan, reset"
    p = figure(
        title="Interactive UMAP Plot with Audio", tools=tools, width=1400, height=800
    )
    # scatter = p.scatter('x', 'y', size=10, source=source)
    scatter = p.scatter("x", "y", size=7, source=source, color=color_mapper)

    # Define CustomJS to handle the tap event
    callback = CustomJS(
        args=dict(source=source),
        code="""
        if (window.currentAudio) {
            window.currentAudio.pause();
            window.currentAudio = null;
        }
        var indices = source.selected.indices;
        if (indices.length > 0) {
            var index = indices[0];
            var audio_file = source.data['absolute_path'][index];
            var audio = new Audio(audio_file);
            audio.play();
            window.currentAudio = audio;
        }
    """,
    )

    # Add the callback to the plot
    p.js_on_event("tap", callback)

    # Add hover tool to the plot
    hover = HoverTool(
        renderers=[scatter],
        tooltips=[
            ("Genre", "@genre"),
            ("Style", "@style"),
            ("File", "@audio_filename"),
        ],
    )
    # Add legend
    # p.legend.title = 'Label'
    p.add_tools(hover)

    # Output to a file (for saving and viewing in a browser)
    output_file(output_filename)

    # Display the plot inline in the notebook
    output_notebook()

    return p


def generate_spectrogram(y, sr, mel_flag, log_flag, win_size, hop, n_mels):
    S = np.abs(librosa.stft(y=y, hop_length=hop, n_fft=win_size)) ** 2
    if mel_flag:
        S = librosa.feature.melspectrogram(S=S, n_mels=n_mels, sr=sr)
    if log_flag:
        S = librosa.power_to_db(S, ref=np.max)

    return S


# def zero_pad(y: np.array, window_size: int):
#     pad_lenght = window_size // 2
#     return np.pad(y, pad_lenght, mode="constant")
