import os.path
import torch.utils.data as data
from data_loading.data_loader.speech_data import Loader
from data_loading.data_split.gcommand_split import audio_partition

def load_partition_data_audio(num_client, folder_path, batch_size, window_size, window_stride, window_type, normalize):

    train_path = os.path.join(folder_path, 'train_training')
    test_path = os.path.join(folder_path, 'train_testing')

    wav_train_data_dict, class_num = audio_partition(num_client, train_path)
    train_data_local_dict = {}
    test_data_local_dict = {idx: None for idx in range(num_client)}
    data_local_num_dict = {}
    train_data_num = 0

    for idx, key in enumerate(wav_train_data_dict):
        train_dataset = Loader(wav_train_data_dict[key], window_size=window_size, window_stride=window_stride,
                               window_type=window_type, normalize=normalize)
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        train_data_local_dict[idx] = train_loader
        train_data_num = train_data_num + len(wav_train_data_dict[key])
        data_local_num_dict[idx] = len(wav_train_data_dict[key])

    wav_global_train, class_num = audio_partition(1, train_path)
    global_train_dataset = Loader(wav_global_train[0], window_size=window_size, window_stride=window_stride,
                           window_type=window_type, normalize=normalize)
    train_data_global = data.DataLoader(dataset=global_train_dataset, batch_size=batch_size, shuffle=True)

    wav_global_test, class_num = audio_partition(1, test_path)
    global_test_dataset = Loader(wav_global_test[0], window_size=window_size, window_stride=window_stride,
                           window_type=window_type, normalize=normalize)
    test_data_global = data.DataLoader(dataset=global_test_dataset, batch_size=batch_size, shuffle=True)
    test_data_num = len(wav_global_test[0])

    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num