# input: wav_data_dict from gcommand_split.py
# task: add fl features (nosiy)
# output: new audio file
import numpy as np
import torch, torchaudio
import os
import sys
from pandas import *
from tqdm import tqdm
from pathlib import Path
from audiomentations import AddBackgroundNoise, PolarityInversion

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from data_split.gcommand_split import audio_partition


def add_noise_snr(audio_file_path, output_path, target_snr_db):

    audio, sample_rate = torchaudio.load(audio_file_path)
    transform_model = torchaudio.transforms.Resample(sample_rate, 16000)
    audio = transform_model(audio)
    # if the audio is not mono channel
    if audio.shape[0] == 2:
        audio = torch.mean(audio, dim=0).unsqueeze(dim=0)
    audio_energy_watts = np.mean(audio.detach().cpu().numpy() ** 2)
    audio_energy_db = 10 * np.log10(audio_energy_watts)

    # calculate noise energy in db
    noise_energy_db = audio_energy_db - target_snr_db
    noise_energy_watts = 10 ** (noise_energy_db / 10)

    # Guassian noise addition
    np.random.seed(8)
    noise = np.random.normal(0, np.sqrt(noise_energy_watts), audio.shape[1])
    noise_audio = audio + noise
    noise_audio = noise_audio.type(torch.float32)
    torchaudio.save(output_path, noise_audio, sample_rate)

def add_esc_snr(audio_file_path, output_path, target_snr_db):

    audio, sample_rate = torchaudio.load(audio_file_path)
    transform_model = torchaudio.transforms.Resample(sample_rate, 16000)
    audio = transform_model(audio)
    # if the audio is not mono channel
    if audio.shape[0] == 2:
        audio = torch.mean(audio, dim=0).unsqueeze(dim=0)
    # read the esc file
    audio = audio.view(audio.shape[1])
    audio = audio.detach().cpu().numpy()
    data = read_csv("../../data/esc_50/ESC-50-master/meta/esc50.csv")
    file_name = data['filename'].tolist()
    file_number = np.random.randint(len(file_name), size=1)
    file_path = "../../data/esc_50/ESC-50-master/audio/" + file_name[file_number[0]]
    waveform, sample_esc_rate = torchaudio.load(str(file_path))
    if waveform.shape[0] != 1:
        waveform = torch.mean(waveform, dim=0).unsqueeze(0)
    if sample_rate != 16000:
        transform_model = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = transform_model(waveform)
    output_esc_path = './' + file_name[file_number[0]]
    torchaudio.save(str(output_esc_path), waveform, 16000)
    # couple with orignial audio
    transform = AddBackgroundNoise(
        sounds_path=output_esc_path,
        min_snr_in_db=target_snr_db,
        max_snr_in_db=target_snr_db,
        noise_transform=PolarityInversion(),
        p=1.0
    )
    noise_audio = transform(audio, sample_rate=16000)
    noise_audio = torch.tensor(noise_audio)
    noise_audio = torch.unsqueeze(noise_audio, 0)
    torchaudio.save(output_path, noise_audio, sample_rate)
    os.remove(output_esc_path)


if __name__ == "__main__":
    # # step 0 train data split
    # folder_path = "/home/ultraz/Project/FedSpeech22/data/speech_commands/train_training"
    # wav_data_dict, class_num = audio_partition(folder_path)
    # # step 1 create fl dataset
    # snr_level = [20, 30, 40]
    # device_ratio = [
    #     round(0.4 * len(wav_data_dict)),
    #     round(0.3 * len(wav_data_dict)),
    #     round(0.3 * len(wav_data_dict)),
    # ]
    # target_snr_db = [0] * len(wav_data_dict)
    # start = 0
    # for i in range(len(device_ratio)):
    #     end = start + device_ratio[i]
    #     target_snr_db[start:end] = [snr_level[i]] * device_ratio[i]
    #     start = end

    # output_folder = "/home/ultraz/Project/FedSpeech22/data/speech_commands/fl_dataset/"
    # Path.mkdir(Path(output_folder), parents=True, exist_ok=True)
    # for i in tqdm(range(len(wav_data_dict))):
    #     for j in range(len(wav_data_dict[i])):
    #         audio_file_path = "../" + wav_data_dict[i][j][1]
    #         output_file_path = output_folder + wav_data_dict[i][j][0] + ".wav"
    #         add_noise_snr(audio_file_path, output_file_path, target_snr_db[i])
    #         wav_data_dict[i][j][1] = output_file_path
    audio_file_path = '/home/ultraz/Project/FedSpeech22/data/iemocap/IEMOCAP_full_release/Session1/dialog/wav/Ses01F_impro01.wav'
    output_path = '/home/ultraz/Project/FedSpeech22/data_loading/fl_feature/test.wav'
    add_esc_snr(audio_file_path, output_path, 20)
