import os.path
import torch
import torch.utils.data as data
from .speech_data import Loader

def pair_dict_gen(folder_path):
    text_path = os.path.join(folder_path, 'text')
    wav_path = os.path.join(folder_path, 'wav.scp')
    segments_path = os.path.join(folder_path, 'segments')

    reader_to_key = dict()
    key_to_word = dict()
    key_to_wav = dict()
    key_to_seg = dict()

    with open(text_path, 'rt') as text:
        for line in text:
            reader, others = line.strip().split('_', 1)
            key, word = line.strip().split(' ', 1)
            if reader not in reader_to_key:
                reader_to_key[reader] = list()
            reader_to_key[reader].append(key)
            key_to_word[key] = word

    with open(wav_path, 'rt') as wav_scp:
        for line in wav_scp:
            key, wav = line.strip().split(' ', 1)
            key_to_wav[key] = '../../' + wav

    if os.path.isfile(segments_path):
        with open(segments_path, 'rt') as segments:
            for line in segments:
                key, wav_key, seg_ini, seg_end = line.strip().split()
                key_to_seg[key] = [wav_key, seg_ini, seg_end]

    return reader_to_key, key_to_word, key_to_wav, key_to_seg

def get_segment(wav, seg_ini, seg_end):
    nwav = None
    if float(seg_end) > float(seg_ini):
        if wav[-1] == '|':
            nwav = wav + ' sox -t wav - -t wav - trim {} ={} |'.format(seg_ini, seg_end)
        else:
            nwav = 'sox {} -t wav - trim {} ={} |'.format(wav, seg_ini, seg_end)
    return nwav

def get_classes():
    CLASSES = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four',
               'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right',
               'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero',
               'silence']
    classes = CLASSES
    weight = None
    class_to_id = {label: i for i, label in enumerate(classes)}
    return classes, weight, class_to_id

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def audio_partition(num_client, folder_path):

    classes, weight, class_to_id = get_classes()
    class_num = len(classes)
    reader_to_key, key_to_word, key_to_wav, key_to_seg = pair_dict_gen(folder_path)
    reader_partition = list(split(list(reader_to_key.keys()), num_client))
    wav_data_dict = dict()
    segments_path = os.path.join(folder_path, 'segments')
    #train_data
    if os.path.isfile(segments_path):
        for idx, i in enumerate(reader_partition):
            if idx not in wav_data_dict:
                wav_data_dict[idx] = list()
            for reader in i:
                keys = reader_to_key[reader]
                for key in keys:
                    wav_key, seg_ini, seg_end = key_to_seg[key]
                    wav_command = key_to_wav[wav_key]
                    word = key_to_word[key]
                    word_id = class_to_id[word]
                    wav_item = [key, get_segment(wav_command, seg_ini, seg_end), word_id]
                    wav_data_dict[idx].append(wav_item)
    #test_data
    else:
        wav_data_dict[0] = list()
        for key, wav_command in key_to_wav.items():
            word = key_to_word[key]
            word_id = class_to_id[word]
            wav_item = [key, wav_command, word_id]
            wav_data_dict[0].append(wav_item)

    return wav_data_dict, class_num


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



if __name__ == '__main__':
    folder_path = "/Users/ultraz/Fed-Mobile/Mobile_Async/data/speech_commands"
    train_path = os.path.join(folder_path, 'train_training')
    test_path = os.path.join(folder_path, 'train_testing')
    reader_to_key, key_to_word, key_to_wav, key_to_seg = pair_dict_gen(train_path)
    train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = load_partition_data_audio(len(reader_to_key), folder_path, 16, .02, .01, 'hamming', True)

    # wav_train_data_dict, class_num = audio_partition(10, train_path)
    # print(wav_train_data_dict[1][1])
    y = 0
    for x in data_local_num_dict.values():
        y = y + x
    print(data_local_num_dict[0])
    print(train_data_num)
    print(len(reader_to_key))
    print(reader_to_key['00176480'])
    print(key_to_seg['00176480_nohash_0_bed'])
    print(key_to_word['00176480_nohash_0_bed'])
