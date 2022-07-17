import os.path
import torch

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
            key_to_wav[key] = '../' + wav

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

def audio_partition(folder_path):

    classes, weight, class_to_id = get_classes()
    class_num = len(classes)
    reader_to_key, key_to_word, key_to_wav, key_to_seg = pair_dict_gen(folder_path)
    num_client = len(reader_to_key.keys())
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
                    wav_item = [key, wav_command, word_id]
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

if __name__ == '__main__':
    #train data
    folder_path = "/home/ultraz/Project/FedSpeech22/data/speech_commands/train_training"
    wav_data_dict, class_num = audio_partition(folder_path)
    print(wav_data_dict[1][0][0])
    #test data
    # folder_path = "/home/ultraz/Project/FedSpeech22/data/speech_commands/train_testing"
    # wav_data_dict, class_num = audio_partition(folder_path)
    # print(wav_data_dict[0])