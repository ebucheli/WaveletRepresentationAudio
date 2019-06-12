import numpy as np
import librosa
from copy import deepcopy
from tqdm import tqdm

def normalize_waveforms(x):
    """Normalize each waveform of a set

    Parameters:
    x: the dataset to normalize

    Returns:
    The same set with each waveform normalized.
    """
    max_nums = np.max(np.absolute(x),axis = 1)
    zeros = np.where(max_nums==0)
    max_nums[zeros] = 1

    return x/max_nums[:,None]


def normalize_spectrograms(x):
    """Normalize each spectrogram of a set

    Parameters:
    x: the dataset to normalize

    Returns:
    The same set with each spectrogram normalized.
    """
    x_2 = np.zeros_like(x)
    for i, example in enumerate(x):
        x_2[i] = example/np.max(np.abs(example))

    return x_2


def normalize_2D(x):
    """Normalize a 2D image
    """

    x_2 = deepcopy(x)
    eps = 1e-10

    means = np.mean(x_2, axis = (1,2))
    std = np.std(x_2,axis = (1,2))

    return (x_2-means[:,None,None])/(std[:,None,None]+eps)


def power_spect_set(x, sr, n_fft, hop_length):
    """Return a set of power spectrograms

    Parameters:
    x: set with waveforms
    sr: Sample Rate
    n_fft: window size for stft
    hop_length: distance between windows for stft

    Returns:
    List with spectrograms
    """
    N = len(x)
    fft_size = int(n_fft/2+1)
    frames = int(np.ceil(sr/hop_length))

    x_2 = np.zeros((N,fft_size,frames))

    for i, wave in enumerate(tqdm(x)):
        spec = np.abs(librosa.core.stft(y = wave,n_fft = n_fft,hop_length=hop_length))
        x_2[i] = librosa.power_to_db(spec)

    return x_2

def mel_spec_set(x, sr, n_mels, hop_length):
    """Return a set of mel spectrograms

    Parameters:
    x: set with waveforms
    sr: Sample Rate
    n_mels: filters in the mel scale
    hop_length: distance between windows for stft

    Returns:
    List with mel spectrograms
    """
    N = len(x)
    frames = int(np.ceil(sr/hop_length))

    x_2 = np.zeros((N, n_mels, frames))

    for i, wave in enumerate(tqdm(x)):
        spec = librosa.feature.melspectrogram(wave,sr = sr,n_mels = n_mels,hop_length = hop_length)
        x_2[i] = librosa.amplitude_to_db(spec)

    return x_2

def mfcc_set(x, sr, n_mfcc, hop_length):
    """Return a set of mel spectrograms

    Parameters:
    x: set with waveforms
    sr: Sample Rate
    n_mfcc: filters in the mel scale
    hop_length: distance between windows for stft

    Returns:
    List with mfccs
    """
    N = len(x)
    frames = int(np.ceil(sr/hop_length))

    x_2 = np.zeros((N, n_mfcc, frames))

    for i, wave in enumerate(tqdm(x)):
        x_2[i] = librosa.feature.mfcc(wave,sr = sr,n_mfcc = n_mfcc,hop_length = hop_length)

    return x_2
