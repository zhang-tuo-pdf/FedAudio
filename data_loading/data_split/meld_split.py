from pathlib import Path

import pandas as pd
from tqdm import tqdm


def audio_partition(folder_path: str, split: str = 'train', task: str = 'sentiment') -> (dict, int):
    """
    Gets wav data dict and the number of unique classes for task.

    :param folder_path: Folder path in which corresponding download script for meld dataset download_audio.sh was executed
    :param split: split to load. Either 'train','test' or 'dev'
    :param task: task to load. Either 'sentiment' or 'emotion'
    :return: data_dict with structure {speaker_index:[[key, file_path, category] ... ] and number of classes
    """
    if split not in ['train', 'test', 'dev']:
        raise ValueError("split must be either 'train','test' or 'dev' for MELD dataset")
    if task not in ['emotion', 'sentiment']:
        raise ValueError("task must be either 'sentiment' or 'emotion' for MELD dataset")
    if split == 'train':
        label_path = f'{folder_path}/train_sent_emo.csv'
        data_path = f'{folder_path}/train_splits'
    elif split == 'test':
        label_path = f'{folder_path}/test_sent_emo.csv'
        data_path = f'{folder_path}/output_repeated_splits_test'
    elif split == 'dev':
        label_path = f'{folder_path}/dev_sent_emo.csv'
        data_path = f'{folder_path}/dev_splits_complete'

    df_label = pd.read_csv(label_path)
    err = []
    for i, df_row in tqdm(df_label.iterrows()):
        if not Path(f"{data_path}/waves/dia{df_row.Dialogue_ID}_utt{df_row.Utterance_ID}.wav").is_file():
            err.append(i)
    print(f'Missing/Corrupt files for indices: {err}')
    df_label_cleaned = df_label.drop(err)
    if task == 'sentiment':
        sentiments = {k: i for i, k in enumerate(df_label_cleaned.Sentiment.unique())}
        df_label_cleaned['Category'] = df_label_cleaned.Sentiment.apply(lambda x: sentiments[x])
        num_classes = len(sentiments.keys())
    elif task == 'emotion':
        emotions = {k: i for i, k in enumerate(df_label_cleaned.Emotion.unique())}
        df_label_cleaned['Category'] = df_label_cleaned.Emotion.apply(lambda x: emotions[x])
        num_classes = len(emotions.keys())

    df_label_cleaned['Path'] = df_label_cleaned.apply(
        lambda row: f"{data_path}/waves/dia{row.Dialogue_ID}_utt{row.Utterance_ID}.wav", axis=1)
    df_label_cleaned['Filename'] = df_label_cleaned.apply(
        lambda row: f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}", axis=1)

    df_label_reduced = df_label_cleaned[['Speaker', 'Filename', 'Path', 'Category']]
    groups = df_label_reduced.groupby('Speaker')
    data_dict = {speaker: group[['Filename', 'Path', 'Category']].values.tolist()
                 for _, (speaker, group) in enumerate(groups) if len(group[['Filename', 'Path', 'Category']]) > 10}
    for filter_speaker in ["All", "Man", "Policeman", "Tag", "Woman"]:
        if filter_speaker in data_dict:
            data_dict.pop(filter_speaker)
    return data_dict, num_classes


if __name__ == '__main__':
    meld_folder_path = "../../data/meld"
    wav_data_dict, class_num = audio_partition(meld_folder_path, 'dev', 'emotion')
    print(wav_data_dict[1][0][0])
