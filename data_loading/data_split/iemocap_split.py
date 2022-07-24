import re, pdb
from tqdm import tqdm
from pathlib import Path


def audio_partition(folder_path, test_session='Session1', split='train'):
    
    class_to_id = {label: i for i, label in enumerate(['neu', 'hap', 'sad', 'ang'])}
    wav_data_dict = dict()
    
    # split the data by speaker
    for session_id in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']:
        ground_truth_path_list = list(Path(folder_path).joinpath(session_id, 'dialog', 'EmoEvaluation').glob('*.txt'))
        ground_truth_path_list.sort()
        for ground_truth_path in tqdm(ground_truth_path_list, ncols=100, miniters=100):
            with open(str(ground_truth_path)) as f: file_content = f.read()
            label_lines = re.findall(re.compile(r'\[.+\]\n', re.IGNORECASE), file_content)
            for line in label_lines:
                if 'Ses' not in line: continue
                if 'impro' not in line: continue
                key = line.split('\t')[-3]
                gender = key.split('_')[-1][0]
                speaker_id = key.split('_')[0][:-1] + gender
                label = line.split('\t')[-2]
                # Four emotion recognition experiments
                if label == 'ang' or label == 'neu' or label == 'sad' or label == 'hap' or label == 'exc':
                    if label == 'exc': label = 'hap'
                    audio_file_path = Path(folder_path).joinpath(session_id, 'sentences', 'wav', '_'.join(key.split('_')[:-1]), key+'.wav')
                    wav_item = [key, str(audio_file_path), class_to_id[label]]
                    if split == 'test':
                        if session_id == test_session: 
                            if speaker_id not in wav_data_dict: wav_data_dict[speaker_id] = list()
                            wav_data_dict[speaker_id].append(wav_item)
                    else:
                        if session_id == test_session: continue
                        if speaker_id not in wav_data_dict: wav_data_dict[speaker_id] = list()
                        wav_data_dict[speaker_id].append(wav_item)
    return wav_data_dict, len(class_to_id)

if __name__ == '__main__':
    folder_path = "/media/data/sail-data/iemocap"
    wav_data_dict, class_num = audio_partition(folder_path)
    print(wav_data_dict[1][0][0])