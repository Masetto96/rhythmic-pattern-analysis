import essentia.standard as es
import warnings
import numpy as np


# def compute_stm_essentia(
#     y: np.ndarray,
#     sr: int,
#     target_sr: int = 8000,
#     mel_flag: bool = True,
#     log_flag: bool = True,
#     detrend: bool = True,
#     win_size: int = 256,
#     hop: int = 160,
#     n_mels: int = 40,
#     with_padding: bool = True,
#     autocor_window_type: str = "hamming",
#     autocor_window_seconds: float = 8,
#     autocor_hop_seconds: float = 0.5,
#     autocor_norm_type: str = "max",
#     autocor_norm_sum: bool = True,
#     num_stm_coefs: int = 200):
#     pass

    # # maybe trim leading and trailing silence?

    # # validate parameters
    # if autocor_window_seconds > (len(y) / sr):  # REVIEW: what to do in this case?
    #     warnings.warn(
    #         "auto_cor_window_seconds is bigger than duration of audio file, setting it to duration"
    #     )
    #     autocor_window_seconds = len(y) / sr

    # if y is None:
    #     raise ValueError("y is not valid")

    # if sr != target_sr:  # TODO: this can be done better if passing flag "with_resample"
    #     y = es.Resample(inputSampleRate = sr, outputSampleRate = target_sr).compute(y)
    #     sr = target_sr

    # windowing = es.Windowing(type='hann')
    # spectrum = es.Spectrum()
    # melbands = es.MelBands(numberBands=n_mels)
    # spectrum_logfreq = es.LogSpectrum()

    # amp2db = es.UnaryOperator(type='lin2db', scale=2)
    # pool = es.Pool()   

    # for frame in es.FrameGenerator(audio, frameSize=win_size, hopSize=hop):

    #     frame_spectrum = spectrum(windowing(frame))
    #     frame_mel = melbands(frame_spectrum)
    #     frame_spectrum_logfreq, _, _ = spectrum_logfreq(frame_spectrum)

    #     pool.add('spectrum_db', amp2db(frame_spectrum))
    #     pool.add('mel_db', amp2db(frame_mel))
    #     pool.add('spectrum_logfreq_db', amp2db(frame_spectrum_logfreq))