from pathlib import Path

import pandas as pd
from tqdm import tqdm


def audio_partition(folder_path, split='train', task='sentiment'):
    if split == 'train':
        label_path = f'{folder_path}/train_sent_emo.csv'
        data_path = f'{folder_path}/train_splits'
    elif split == 'test':
        label_path = f'{folder_path}/MELD.Raw/test_sent_emo.csv'
        data_path = f'{folder_path}/output_repeated_splits_test'
    elif split == 'dev':
        label_path = f'{folder_path}/MELD.Raw/dev_sent_emo.csv'
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
        lambda row: f"{data_path}/waves/dia{df_row.Dialogue_ID}_utt{df_row.Utterance_ID}.wav", axis=1)

    df_label_reduced = df_label_cleaned[['Speaker', 'Path', 'Category']]
    groups = df_label_reduced.groupby('Speaker')
    wav_data_dict = {i: group[['Speaker', 'Path', 'Category']].values.tolist()
                     for i, (speaker, group) in enumerate(groups)}
    return wav_data_dict, num_classes


if __name__ == '__main__':
    folder_path = "../../data/meld/"
    wav_data_dict, class_num = audio_partition(folder_path, 'dev', 'emotion')
    print(wav_data_dict[1][0][0])