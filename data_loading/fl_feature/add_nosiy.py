# input: wav_data_dict from gcommand_split.py
# task: add fl features (nosiy)
# output: new audio file
import numpy as np
import copy, pdb
import torch, torchaudio

def add_noise_snr(audio_file_path, output_path, target_snr_db):
    
    audio, sample_rate = torchaudio.load(audio_file_path)
    transform_model = torchaudio.transforms.Resample(sample_rate, 16000)
    audio = transform_model(audio)

    audio_energy_watts = np.mean(audio.detach().cpu().numpy() ** 2)
    audio_energy_db = 10 * np.log10(audio_energy_watts)
    
    # calculate noise energy in db
    noise_energy_db = audio_energy_db - target_snr_db
    noise_energy_watts = 10 ** (noise_energy_db / 10)
    
    # Guassian noise addition
    noise = np.random.normal(0, np.sqrt(noise_energy_watts), len(audio))
    noise_audio = audio + noise
    noise_audio = noise_audio.type(torch.float32)
    pdb.set_trace()
    torchaudio.save(output_path, noise_audio, sample_rate)

if __name__ == '__main__':
    #train data
    audio_file_path = "/home/ultraz/Project/FedSpeech22/data/speech_commands/audio/bed/0a7c2a8d_nohash_0.wav"
    output_path = "/home/ultraz/Project/FedSpeech22/data/"
    file_name = ['test1.wav', 'test10.wav', 'test50.wav', 'test100.wav']
    target_snr_db = [1, 10, 50, 100]
    for i in range(len(file_name)):
        output_file_path = output_path + file_name[i]
        add_noise_snr(audio_file_path, output_file_path, target_snr_db[i])