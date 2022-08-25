from builtins import print
import torchaudio, torch, argparse, pdb, pickle
import numpy as np
from tqdm import tqdm


def mfcc(audio_file_path):

    audio_transform = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=40,
        melkwargs={"n_fft": 800, "hop_length": 160, "power": 2},
    )
    audio, sample_rate = torchaudio.load(audio_file_path)
    transform_model = torchaudio.transforms.Resample(sample_rate, 16000)
    audio = transform_model(audio)

    mfcc = audio_transform(audio).detach()

    der1 = np.expand_dims(np.gradient(audio[0]), axis=0)
    der2 = np.expand_dims(np.gradient(audio[0], 2), axis=0)

    delta = audio_transform(torch.from_numpy(der1)).detach()
    ddelta = audio_transform(torch.from_numpy(der2)).detach()

    return np.concatenate((mfcc, delta, ddelta), axis=1)


def mel_spectrogram(audio_file_path, n_fft=1024, feature_len=128):

    window_size = n_fft
    window_hop = 160
    n_mels = feature_len
    window_fn = torch.hann_window

    audio, sample_rate = torchaudio.load(audio_file_path)
    transform_model = torchaudio.transforms.Resample(sample_rate, 16000)
    audio = transform_model(audio)

    audio_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=int(window_size),
        hop_length=int(window_hop),
        window_fn=window_fn,
    )
    # print(audio_file_path)
    audio_amp_to_db = torchaudio.transforms.AmplitudeToDB()
    return audio_amp_to_db(audio_transform(audio).detach())[0].cpu().numpy().T


if __name__ == "__main__":
    audio_file_path = "/home/ultraz/Project/FedSpeech22/data/speech_commands/audio/bed/0a7c2a8d_nohash_0.wav"
    features = mfcc(audio_file_path)
    print(features)
    features = mel_spectrogram(audio_file_path)
    print(features)
