import re, pdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import KFold

def audio_partition(folder_path, test_fold=1, split='train'):
    # get speaker folds
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    speaker_fold = [speaker_fold for speaker_fold in kf.split(np.arange(1001, 1092, 1))][test_fold-1]
    speaker_ids = speaker_fold[0] if split == 'train' else speaker_fold[1]
    class_to_id = {label: i for i, label in enumerate(['N', 'H', 'S', 'A'])}
    
    wav_data_dict = dict()
    
    # read ground truth annotations
    rating_df = pd.read_csv(str(Path(folder_path).joinpath('processedResults', 'summaryTable.csv')), index_col=1)
    
    # split the data by speaker
    for speaker_id in speaker_ids:
        file_list = list(Path(folder_path).joinpath('AudioWAV').glob("10"+str(speaker_id)+'_*.wav'))
        for file_name in file_list:
            label = rating_df.loc[str(file_name).split('/')[-1].split('.wav')[0], 'MultiModalVote']
            # correputed audio file per download
            if '1076_MTI_SAD_XX.wav' in str(file_name): continue
            # if emotion is not in 4-emotion category, we skip
            if label not in class_to_id: continue
            # initialize wav_data_dict
            if speaker_id not in wav_data_dict: wav_data_dict[speaker_id] = list()
            wav_item = [speaker_id, str(file_name), class_to_id[label]]
            wav_data_dict[speaker_id].append(wav_item)
    return wav_data_dict, len(class_to_id)

if __name__ == '__main__':
    folder_path = "/media/data/public-data/crema-d"
    wav_data_dict, class_num = audio_partition(folder_path)
    print(wav_data_dict[1][0][0])