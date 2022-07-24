import re, pdb
from tqdm import tqdm
from pathlib import Path


def audio_partition(folder_path, test_session='session1', split='train'):
    class_to_id = {label: i for i, label in enumerate(['N', 'H', 'S', 'A'])}
    wav_data_dict = dict()
    
    # read ground truth file
    evaluation_path = Path(folder_path).joinpath('Evalution.txt')
    with open(str(evaluation_path)) as f:
        evaluation_lines = f.readlines()
    # read to a label dict
    label_dict = {}
    for evaluation_line in evaluation_lines:
        if 'UTD-' not in evaluation_line: continue
        audio_file_name = 'MSP-'+evaluation_line.split('.avi')[0][4:]
        label_dict[audio_file_name] = evaluation_line.split('; ')[1][0]
    # read data to desired structure
    for session_id in ['session1', 'session2', 'session3', 'session4', 'session5', 'session6']:
        if split == 'train' and test_session == session_id: continue
        if split == 'test' and session_id != test_session: continue
        audio_file_paths = list(Path(folder_path).joinpath('Audio', session_id).glob('*/*/*.wav'))
        for audio_file_path in audio_file_paths:
            audio_file_name = str(audio_file_path).split("/")[-1].split(".wav")[0]
            audio_file_part = audio_file_name.split('-')
            # read speaker id, label, recording type
            speaker_id = audio_file_part[-3]
            label = label_dict[audio_file_name]
            recording_type = audio_file_part[-2][-1:]
            # we keep improv data only
            if recording_type == 'P' or recording_type == 'R': continue
            if label not in class_to_id: continue
            wav_item = [speaker_id, str(audio_file_path), class_to_id[label]]
            if speaker_id not in wav_data_dict: wav_data_dict[speaker_id] = list()
            wav_data_dict[speaker_id].append(wav_item)
    return wav_data_dict, len(class_to_id)

if __name__ == '__main__':
    folder_path = "/media/data/sail-data/"
    wav_data_dict, class_num = audio_partition(folder_path)
    print(wav_data_dict[1][0][0])