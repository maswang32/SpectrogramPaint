import torch
import torchaudio
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

SAMPLE_RATE = 44100
N_FFT = 17640
HOP_LENGTH = 441
WIN_LENGTH = 4410
F_MIN = 0
F_MAX = 10000
N_MELS = 512

spectrogram_func = torchaudio.transforms.Spectrogram(
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    pad=0,
    window_fn=torch.hann_window,
    power=None,
    normalized=False,
    wkwargs=None,
    center=True,
    pad_mode="reflect",
    onesided=True,
).to(device)

mel_scaler = torchaudio.transforms.MelScale(
    n_mels=N_MELS,
    sample_rate=SAMPLE_RATE,
    f_min=F_MIN,
    f_max=F_MAX,
    n_stft = N_FFT//2 + 1,
    norm = None,
    mel_scale = "htk"
).to(device)

inverse_spectrogram_func = torchaudio.transforms.GriffinLim(
    n_fft=N_FFT,
    n_iter=32,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    window_fn=torch.hann_window,
    power=1.0,
    wkwargs=None,
    momentum=0.9,
    length=None,
    rand_init=True,
).to(device)

inverse_mel_scaler = torchaudio.transforms.InverseMelScale(
    n_stft=N_FFT // 2 + 1,
    n_mels=N_MELS,
    sample_rate=SAMPLE_RATE,
    f_min=F_MIN,
    f_max=F_MAX,
    norm=None,
    mel_scale="htk",
).to(device)

def waveform_from_spectrogram(mel_amplitudes, filter=True):
    mel_amplitudes_torch = torch.from_numpy(mel_amplitudes).to(device)
    amplitudes_linear = inverse_mel_scaler(mel_amplitudes_torch)
    waveform = inverse_spectrogram_func(amplitudes_linear)
    waveform = waveform.detach().cpu().numpy()
    return waveform