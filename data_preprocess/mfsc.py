from functools import lru_cache
import numpy as np


@lru_cache(maxsize=10)
def get_window(n, type='hamming'):
    coefs = np.arange(n)
    window = 0.54 - 0.46 * np.cos(2 * np.pi * coefs / (n - 1))
    return window


def apply_preemphasis(y, preemCoef=0.97):
    y[1:] = y[1:] - preemCoef * y[:-1]
    y[0] *= (1 - preemCoef)
    return y


def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def mel_to_freq(mels):
    return 700.0 * (np.power(10.0, mels / 2595.0) - 1.0)


@lru_cache(maxsize=10)
def get_filterbank(numfilters, filterLen, lowFreq, highFreq, samplingFreq):
    minwarpfreq = freq_to_mel(lowFreq)
    maxwarpfreq = freq_to_mel(highFreq)
    dwarp = (maxwarpfreq - minwarpfreq) / (numfilters + 1)
    f = mel_to_freq(np.arange(numfilters + 2) * dwarp + minwarpfreq) * (filterLen - 1) * 2.0 / samplingFreq
    i = np.arange(filterLen)[None, :]
    f = f[:, None]
    hislope = (i - f[:numfilters]) / (f[1:numfilters + 1] - f[:numfilters])
    loslope = (f[2:numfilters + 2] - i) / (f[2:numfilters + 2] - f[1:numfilters + 1])
    H = np.maximum(0, np.minimum(hislope, loslope))
    return H


def normalized(y, threshold=0):
    y -= y.mean()
    stddev = y.std()
    if stddev > threshold:
        y /= stddev
    return y


def mfsc(y, sfr, window_size=0.025, window_stride=0.010, window='hamming', normalize=True, log=True, n_mels=80,
         preemCoef=0.97, melfloor=1.0):
    win_length = int(sfr * window_size)
    hop_length = int(sfr * window_stride)
    n_fft = 512
    lowfreq = 0
    highfreq = sfr / 2

    # get window
    window = get_window(win_length)
    padded_window = np.pad(window, (0, n_fft - win_length), mode='constant')[:, None]

    # preemphasis
    y = apply_preemphasis(y, preemCoef)

    # scale wave signal
    y *= 32768

    # get frames and scale input
    num_frames = 1 + (len(y) - win_length) // hop_length
    pad_after = num_frames * hop_length + (n_fft - hop_length) - len(y)
    if pad_after > 0:
        y = np.pad(y, (0, pad_after), mode='constant')
    frames = np.lib.stride_tricks.as_strided(y, shape=(n_fft, num_frames),
                                             strides=(y.itemsize, hop_length * y.itemsize), writeable=False)
    windowed_frames = padded_window * frames
    D = np.abs(np.fft.rfft(windowed_frames, axis=0))

    # mel filterbank
    filterbank = get_filterbank(n_mels, n_fft / 2 + 1, lowfreq, highfreq, sfr)
    mf = np.dot(filterbank, D)
    mf = np.maximum(melfloor, mf)
    if log:
        mf = np.log(mf)
    if normalize:
        mf = normalized(mf)

    return mf
