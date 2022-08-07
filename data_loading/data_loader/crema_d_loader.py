import os.path
import sys
import logging
import pickle
import argparse
import pdb
from tqdm import tqdm
from pathlib import Path
import torch.utils.data as data
from wandb import set_trace
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from speech_data import DatasetGenerator, collate_fn_padd
from data_split.crema_d_split import audio_partition
from fl_feature.add_nosiy import add_noise_snr
from data_preprocess.opensmile_extractor import opensmile_feature
from data_preprocess.pretrain_model_extractor import pretrained_feature, load_model
from data_preprocess.raw_audio_process import mel_spectrogram
from data_preprocess.speaker_normalization import speaker_normalization


def load_partition_data_audio(
    raw_data_path,
    output_path,
    batch_size,
    process_method,
    feature_type=None,
    fl_feature=None,
    snr_level=None,
    device_ratio=None,
):
    # raw data root path
    folder_path = Path(raw_data_path)
    
    # train dataset
    logging.info("data split begin")
    wav_train_data_dict, class_num = audio_partition(folder_path, test_fold=args.test_fold, split='train')
    logging.info("data split finish")
    client_idx = list(wav_train_data_dict.keys())

    # fl feature: noise addition
    if fl_feature:
        logging.info("add federated learning related features")
        output_folder = (
            os.path.join(output_path, "fl_dataset/")
        )
        if os.path.isdir(output_folder):
            shutil.rmtree(output_folder)        
        if not os.path.isdir(output_folder):
            # step 1 create fl dataset
            logging.info("create federated learning dataset")
            target_snr_db = [0] * len(wav_train_data_dict)
            start = 0
            for i in range(len(device_ratio)):
                end = start + round(device_ratio[i] * len(wav_train_data_dict))
                #print("this is ", round(device_ratio[i] * len(wav_train_data_dict)))
                target_snr_db[start:end] = [snr_level[0]] * round(device_ratio[i] * len(wav_train_data_dict))
                start = end
            Path.mkdir(Path(output_folder), parents=True, exist_ok=True)
            for i in tqdm(range(len(wav_train_data_dict))):
                for j in range(len(wav_train_data_dict[client_idx[i]])):
                    audio_file_path = wav_train_data_dict[client_idx[i]][j][1]
                    a = audio_file_path.split('/', 9 )
                    output_file_path = (
                        output_folder + a[6]
                    )
                    add_noise_snr(audio_file_path, output_file_path, target_snr_db[i])
                    wav_train_data_dict[client_idx[i]][j][1] = output_file_path
        else:
            for i in tqdm(range(len(wav_train_data_dict))):
                for j in range(len(wav_train_data_dict[client_idx[i]])):
                    audio_file_path = wav_train_data_dict[client_idx[i]][j][1]
                    a = audio_file_path.split('/', 9 )
                    output_file_path = (
                        output_folder + a[6]
                    )
                    wav_train_data_dict[client_idx[i]][j][1] = output_file_path
    logging.info("federated learning feature loaded")
    
    # step 2 preprocess data
    logging.info("begin data preprocess")
    if process_method == "pretrain":
        device, model = load_model(feature_type)
    for i in tqdm(wav_train_data_dict):
        for j in range(len(wav_train_data_dict[i])):
            audio_file_path = wav_train_data_dict[i][j][1]
            if process_method == "opensmile_feature":
                features = opensmile_feature(audio_file_path, feature_type)
            elif process_method == "pretrain":
                features = pretrained_feature(
                    audio_file_path, feature_type, device, model
                )
            elif process_method == "raw":
                features = mel_spectrogram(audio_file_path)
            wav_train_data_dict[i][j].append(features)
    logging.info("data have been processed")
    
    # train local data and test local data
    train_data_local_dict = {}
    test_data_local_dict = {idx: None for idx in range(len(wav_train_data_dict))}
    data_local_num_dict = {}
    train_data_num = 0

    logging.info("loading local training data")
    for idx, key in enumerate(wav_train_data_dict):
        train_dataset = DatasetGenerator(speaker_normalization(wav_train_data_dict[key]))
        train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_padd,
        )
        train_data_local_dict[idx] = train_loader
        train_data_num = train_data_num + len(wav_train_data_dict[key])
        data_local_num_dict[idx] = len(wav_train_data_dict[key])
    logging.info("finish data loading")
    train_data_global = None

    # test dataset
    wav_global_test, class_num = audio_partition(folder_path, test_fold=args.test_fold, split='test')
    # step 2 preprocess data
    logging.info("begin test data preprocess")
    if process_method == "pretrain":
        device, model = load_model(feature_type)
    for i in tqdm(wav_global_test):
        for j in range(len(wav_global_test[i])):
            audio_file_path = wav_global_test[i][j][1]
            if process_method == "opensmile_feature":
                features = opensmile_feature(audio_file_path, feature_type)
            elif process_method == "pretrain":
                features = pretrained_feature(
                    audio_file_path, feature_type, device, model
                )
            elif process_method == "raw":
                features = mel_spectrogram(audio_file_path)
            wav_global_test[i][j].append(features)

    # save to 0 key
    wav_test = []
    for i in tqdm(wav_global_test):
        wav_global_test[i] = speaker_normalization(wav_global_test[i])
        for j in range(len(wav_global_test[i])):
            wav_test.append(wav_global_test[i][j])
    
    logging.info("test data have been processed")
    global_test_dataset = DatasetGenerator(wav_test)
    test_data_global = data.DataLoader(
        dataset=global_test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_padd,
    )
    test_data_num = len(wav_test)

    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )


if __name__ == "__main__":
    # step 0 train data split
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--raw_data_path', type=str, default='../../data/crema_d/CREMA-D', help='Raw data path of Crema-D data set'
    )
    
    parser.add_argument(
        '--output_data_path', type=str, default='../../data/crema_d', help='Output path of Crema-D data set'
    )
    
    parser.add_argument(
        '--process_method', type=str, default='pretrain', help='Process method: pretrain; raw; opensmile_feature'
    )
    
    parser.add_argument(
        '--feature_type', type=str, default='apc', help='Feature type based on the process_method method'
    )
    
    parser.add_argument(
        '--test_fold', type=int, default=1, help='Test fold id for Crema-D dataset, default test fold is 1'
    )

    parser.add_argument(
        "--db_level",
        type=float,
        default=20,
        help="db level for the adding nosiy",
    )

    args = parser.parse_args()
    
    if not Path(args.raw_data_path).exists(): 
        raise Exception("Crema-D data not found, please check arg raw_data_path!")
    if args.test_fold not in [1, 2, 3, 4, 5]:
        raise Exception("Invailid test fold, available options: 1; 2; 3; 4; 5.")
    Path(args.output_data_path).mkdir(parents=True, exist_ok=True)
    
    batch_size = 16
    fl_feature = True
    # snr_level = [20, 30, 40]
    snr_level = [args.db_level]
    #device_ratio = [round(0.4 * 2118), round(0.3 * 2118), round(0.3 * 2118)]
    device_ratio = [0.4, 0.3, 0.3]
    (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = load_partition_data_audio(
        args.raw_data_path,
        args.output_data_path,
        batch_size,
        args.process_method,
        feature_type=args.feature_type,
        fl_feature=fl_feature,
        snr_level=snr_level,
        device_ratio=device_ratio,
    )
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    if fl_feature == True:
        save_file_name = 'processed_dataset_'+args.process_method+'_'+args.feature_type+'_fold_'+str(args.test_fold)+"_db"+str(args.db_level)+'.p'
    else:
        save_file_name = 'processed_dataset_'+args.process_method+'_'+args.feature_type+'_fold_'+str(args.test_fold)+'.p'
    save_data_path = Path(args.output_data_path).joinpath(save_file_name)
    pickle.dump(dataset, open(save_data_path, "wb"))
    print('data finished')

#taskset 100 python
