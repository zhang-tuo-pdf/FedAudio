import os.path
import subprocess
import struct
import wave

import numpy as np
from .mfsc import mfsc
import torch
import torch.utils.data as data

try:
    from subprocess import DEVNULL # python3
except ImportError:
    DEVNULL = open(os.devnull, 'wb')

def wav_read(pipe):
    if pipe[-1] == '|':
        tpipe = subprocess.Popen(pipe[:-1], shell=True, stderr=DEVNULL, stdout=subprocess.PIPE)
        audio = tpipe.stdout
    else:
        tpipe = None
        audio = pipe
    try:
        wav = wave.open(audio, 'r')
    except EOFError:
        print('EOFError:', pipe)
        exit(-1)
    sfreq = wav.getframerate()
    assert wav.getsampwidth() == 2
    wav_bytes = wav.readframes(-1)
    npts = len(wav_bytes) // wav.getsampwidth()
    wav.close()
    # convert binary chunks
    wav_array = np.array(struct.unpack("%ih" % npts, wav_bytes), dtype=float) / (1 << 15)
    return wav_array, sfreq

def param_loader(path, window_size, window_stride, window, normalize, max_len):
    y, sfr = wav_read(path)

    param = mfsc(y, sfr, window_size=window_size, window_stride=window_stride, window=window, normalize=normalize, log=False, n_mels=40, preemCoef=0, melfloor=1.0)

    # Add zero padding to make all param with the same dims
    if param.shape[1] < max_len:
        pad = np.zeros((param.shape[0], max_len - param.shape[1]))
        param = np.hstack((pad, param))

    # If exceeds max_len keep last samples
    elif param.shape[1] > max_len:
        param = param[:, -max_len:]

    param = torch.FloatTensor(param)

    return param

class Loader(data.Dataset):
    """A google commands data set loader using Kaldi data format::
    Args:
        root (string): Kaldi directory path.
        transform (callable, optional): A function/transform that takes in a spectrogram
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the param to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        classes (list): List of the class names.
        class_to_id (dict): Dict with items (class_name, class_index).
        wavs (list): List of (wavs path, class_index) tuples
        STFT parameters: window_size, window_stride, window_type, normalize
    """

    def __init__(self, wavs, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=99):

        self.wavs = wavs
        self.weight = None
        self.transform = transform
        self.target_transform = target_transform
        self.loader = param_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (params, target) where target is class_index of the target class.
        """
        key, path, target = self.wavs[index]
        params = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)  # pylint: disable=line-too-long
        if self.transform is not None:
            params = self.transform(params)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return key, params, target

    def __len__(self):
        return len(self.wavs)

if __name__ == '__main__':
    path = 'sox ../data/speech_commands/audio/bed/00176480_nohash_0.wav -t wav - trim 0 =1 |'
    params = param_loader(path, .02, .01, 'hamming', True, 99)
    print(params)