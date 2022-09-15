import numpy as np
import copy


def speaker_normalization(data_dict):
    # append all data
    return_dict = copy.deepcopy(data_dict)
    speaker_data = np.array(data_dict[0][-1]).copy()
    for idx in range(1, len(data_dict)):
        speaker_data = np.append(speaker_data, np.array(data_dict[idx][-1].copy()), axis=0)
    speaker_mean, speaker_std = np.mean(speaker_data, axis=0), np.std(
        speaker_data, axis=0
    )
    # calculate normalized data
    for idx in range(len(return_dict)):
        return_dict[idx][-1] = (data_dict[idx][-1].copy() - speaker_mean) / (speaker_std + 1e-5)
    return return_dict
