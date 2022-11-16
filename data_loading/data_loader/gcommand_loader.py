import os.path
import sys
import logging
import pickle
from tqdm import tqdm
from pathlib import Path
import argparse
import shutil
import numpy as np
import torch.utils.data as data

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from speech_data import DatasetGenerator, collate_fn_padd
from data_split.gcommand_split import audio_partition
from fl_feature.add_nosiy import add_noise_snr
from data_preprocess.opensmile_extractor import opensmile_feature
from data_preprocess.pretrain_model_extractor import pretrained_feature, load_model
from data_preprocess.raw_audio_process import mfcc, mel_spectrogram


def load_partition_data_audio(
    batch_size,
    process_method,
    feature_type=None,
    fl_feature=None,
    snr_level=None,
    setup=None,
    device_ratio=None,
):

    folder_path = os.path.join(os.getcwd(), "..", "..", "data/speech_commands")
    train_path = os.path.join(folder_path, "train_training")
    test_path = os.path.join(folder_path, "train_testing")

    # train dataset
    logging.info("data split begin")
    wav_train_data_dict, class_num = audio_partition(train_path)
    # if setup is centralized, there is no clients
    if setup == "centralized":
        wav_train_data_dict[0] = list()
        for key in wav_train_data_dict:
            if key == 0:
                continue
            for idx in range(len(wav_train_data_dict[key])):
                wav_train_data_dict[0].append(wav_train_data_dict[key][idx])
        key_list = list(wav_train_data_dict.keys())
        for key in key_list:
            if key != 0: 
                wav_train_data_dict.pop(key)
                
    logging.info("data split finish")
    if fl_feature:
        logging.info("add federated learning related features")
        output_folder = os.path.join(folder_path, "fl_dataset/")
        if os.path.isdir(output_folder):
            shutil.rmtree(output_folder)   
        if not os.path.isdir(output_folder):
            # step 1 create fl dataset
            logging.info("create federated learning dataset")
            target_snr_db = [0] * len(wav_train_data_dict)
            start = 0
            for i in range(len(device_ratio)):
                end = start + device_ratio[i]
                target_snr_db[start:end] = [snr_level[i]] * device_ratio[i]
                start = end
            Path.mkdir(Path(output_folder), parents=True, exist_ok=True)
            for i in tqdm(range(len(wav_train_data_dict))):
                for j in range(len(wav_train_data_dict[i])):
                    audio_file_path = "../" + wav_train_data_dict[i][j][1]
                    output_file_path = (
                        output_folder + wav_train_data_dict[i][j][0] + ".wav"
                    )
                    add_noise_snr(audio_file_path, output_file_path, target_snr_db[i])
                    wav_train_data_dict[i][j][1] = output_file_path
        else:
            for i in tqdm(range(len(wav_train_data_dict))):
                for j in range(len(wav_train_data_dict[i])):
                    output_file_path = (
                        output_folder + wav_train_data_dict[i][j][0] + ".wav"
                    )
                    wav_train_data_dict[i][j][1] = output_file_path
    logging.info("federated learning feature loaded")

    # step 2 preprocess data
    logging.info("begin data preprocess")
    if process_method == "pretrain":
        device, model = load_model(feature_type)
    for i in tqdm(range(len(wav_train_data_dict))):
        for j in range(len(wav_train_data_dict[i])):
            if fl_feature:
                audio_file_path = wav_train_data_dict[i][j][1]
            else:
                audio_file_path = "../" + wav_train_data_dict[i][j][1]
            if process_method == "opensmile_feature":
                features = opensmile_feature(audio_file_path, feature_type)
            elif process_method == "pretrain":
                features = pretrained_feature(
                    audio_file_path, feature_type, device, model
                )
            elif process_method == "raw":
                features = mel_spectrogram(audio_file_path)
            features =  (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-5)
            wav_train_data_dict[i][j].append(features)
    logging.info("data have been processed")
    train_data_local_dict = {}
    test_data_local_dict = {idx: None for idx in range(len(wav_train_data_dict))}
    data_local_num_dict = {}
    train_data_num = 0

    logging.info("loading local training data")
    for idx, key in enumerate(wav_train_data_dict):
        train_dataset = DatasetGenerator(wav_train_data_dict[key])
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
    wav_global_test, class_num = audio_partition(test_path)
    # step 2 preprocess data
    logging.info("begin test data preprocess")
    if process_method == "pretrain":
        device, model = load_model(feature_type)
    for i in range(len(wav_global_test)):
        for j in tqdm(range(len(wav_global_test[i]))):
            # audio_file_path = wav_global_test[i][j][1]
            audio_file_path = "../" + wav_global_test[i][j][1]
            if process_method == "opensmile_feature":
                features = opensmile_feature(audio_file_path, feature_type)
            elif process_method == "pretrain":
                features = pretrained_feature(
                    audio_file_path, feature_type, device, model
                )
            elif process_method == "raw":
                features = mel_spectrogram(audio_file_path)
            features =  (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-5)
            wav_global_test[i][j].append(features)
    logging.info("test data have been processed")
    global_test_dataset = DatasetGenerator(wav_global_test[0])
    test_data_global = data.DataLoader(
        dataset=global_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_padd,
    )
    test_data_num = len(wav_global_test[0])

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
        "--raw_data_path",
        type=str,
        default="../../data/speech_commands/audio",
        help="Raw data path of speech_commands data set",
    )

    parser.add_argument(
        "--output_data_path",
        type=str,
        default="../../data/speech_commands",
        help="Output path of speech_commands data set",
    )

    parser.add_argument(
        "--process_method",
        type=str,
        default="raw",
        help="Process method: pretrain; raw; opensmile_feature",
    )
    
    parser.add_argument(
        "--fl_feature",
        default=False,
        type=lambda x: (str(x).lower() == 'true'),
        help="Adding Federated features or not: True/False"
    )

    parser.add_argument(
        "--feature_type",
        type=str,
        default="mel_spec",
        help="Feature type based on the process_method method",
    )

    parser.add_argument(
        "--db_level",
        type=float,
        default=30,
        help="db level for the adding nosiy",
    )
    
    parser.add_argument(
        "--setup",
        type=str,
        default="federated",
        help="setup of the experiment: centralized/federated",
    )
    args = parser.parse_args()

    batch_size = 16
    if args.setup != "federated":
        batch_size = 64

    fl_feature = args.fl_feature
    # snr_level = [20, 20, 20]
    # device_ratio = [round(0.4 * 2118), round(0.3 * 2118), round(0.3 * 2118)]
    snr_level = [args.db_level]
    device_ratio = [2118]
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
        batch_size,
        args.process_method,
        feature_type=args.feature_type,
        setup=args.setup,
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
        save_file_name = (
            args.setup+"_dataset_" + args.process_method + "_" + args.feature_type + "_db" + str(args.db_level) + ".p"
        )
    else:
        save_file_name = (
            args.setup+"_dataset_" + args.process_method + "_" + args.feature_type + ".p"
        )
    save_data_path = Path(args.output_data_path).joinpath(save_file_name)
    pickle.dump(dataset, open(save_data_path, "wb"))
    print("data finished")
