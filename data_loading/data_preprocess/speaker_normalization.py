import numpy as np

def speaker_normalization(data_dict):
    # append all data
    speaker_data = np.array(data_dict[0][-1])
    for idx in range(1, len(data_dict)):
        speaker_data = np.append(speaker_data, np.array(data_dict[idx][-1]), axis=0)
    speaker_mean, speaker_std = np.mean(speaker_data, axis=0), np.std(speaker_data, axis=0)
    # calculate normalized data
    for idx in range(len(data_dict)):
        data_dict[idx][-1] = (data_dict[idx][-1] - speaker_mean) / (speaker_std + 1e-5)
    return data_dict