import re, pdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import KFold

def audio_partition(folder_path, test_fold=1, split='train', num_clients=50):
    # get audio files
    audio_file_paths = list(Path(folder_path).joinpath('audio').glob('*/*.wav'))
    unique_file_dict = dict()
    for audio_file_path in audio_file_paths:
        audio_file_fold = str(audio_file_path).split("/")[-2]
        # train split and test fold, skip the data
        if "fold"+str(test_fold) == audio_file_fold and split == 'train':
            continue
        # test split and train fold, skip the data
        if "fold"+str(test_fold) != audio_file_fold and split == 'test':
            continue
        audio_file_name = str(audio_file_path).split("/")[-1].split(".wav")[0]
        file_id = audio_file_name.split("-")[0]
        class_id = audio_file_name.split("-")[1]
        wav_item = [file_id, str(audio_file_path), int(class_id)]
        
        if file_id not in unique_file_dict: unique_file_dict[file_id] = list()
        unique_file_dict[file_id].append(wav_item)
    
    # split the data via direchlet
    file_label_list = [unique_file_dict[file_id][0][-1] for file_id in unique_file_dict]
    file_id_list = [file_id for file_id in unique_file_dict]
    wav_data_dict = dict()
    
    if split == 'train':
        # cut the data using dirichlet
        min_size = 0
        K, N = len(np.unique(file_label_list)), len(file_label_list)
        # at least we train 1 full batch
        min_sample_size = 5
        np.random.seed(0)
        while min_size < min_sample_size:
            file_idx_clients = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(np.array(file_label_list) == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(0.1, num_clients))
                # Balance
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p, idx_j in zip(proportions, file_idx_clients)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                file_idx_clients = [idx_j + idx.tolist() for idx_j,idx in zip(file_idx_clients,np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in file_idx_clients])

        # save to wav_data_dict
        for client_idx in range(num_clients):
            wav_data_dict[client_idx] = list()
            select_file_ids = [file_id_list[idx] for idx in file_idx_clients[client_idx]]
            for file_id in select_file_ids:
                for data in unique_file_dict[file_id]:
                    wav_data_dict[client_idx].append(data)
    else:
        wav_data_dict[0] = list()
        for file_id in unique_file_dict:
            for data in unique_file_dict[file_id]:
                wav_data_dict[0].append(data)
        
    return wav_data_dict, 10

if __name__ == '__main__':
    folder_path = "/media/data/public-data/crema-d"
    wav_data_dict, class_num = audio_partition(folder_path)
    print(wav_data_dict[1][0][0])