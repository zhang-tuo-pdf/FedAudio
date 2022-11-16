import argparse
import logging
from pathlib import Path
import os
import random
import socket
import sys
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import wandb
from tqdm import tqdm

# add the FedML root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../data_loading")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.getcwd(), "../data_loading/data_loader"))
)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.getcwd(), "../data_loading/data_preprocess"))
)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.getcwd(), "../data_loading/data_split"))
)
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from data_loading.data_loader import global_constant
from data_loading.data_loader.speech_data import DatasetGenerator, collate_fn_padd
from model.vgg_speech import VGG
from model.LeNet import LeNet
from model.bc_resnet import BCResNet
from model.conv_model import audio_conv_rnn
from model.conv_model import audio_rnn
from trainers.speech_trainer import MyModelTrainer
from trainers.fedprox_speech_trainer import FedProxModelTrainer
from FedML.fedml_api.distributed.fedavg.FedAvgAPI import (
    FedML_init,
    FedML_FedAvg_distributed,
)
from FedML.fedml_api.distributed.fedavg_seq.FedAvgSeqAPI import (
    FedML_init,
    FedML_FedAvgSeq_distributed,
)
from FedML.fedml_api.distributed.fedopt_seq.FedOptSeqAPI import FedML_FedOptSeq_distributed
from FedML.fedml_api.distributed.fedopt.FedOptAPI import FedML_FedOpt_distributed
from FedML.fedml_api.distributed.fedprox.FedProxAPI import FedML_FedProx_distributed


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument(
        "--model",
        type=str,
        default="audio_conv_rnn",
        metavar="N",
        help="neural network used in training",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="gcommand",
        metavar="N",
        help="dataset used for training",
    )

    parser.add_argument(
        "--data_dir", type=str, default="../data/", help="data directory"
    )

    parser.add_argument(
        "--client_num_in_total",
        type=int,
        default=2118,
        metavar="NN",
        help="number of workers in a distributed cluster",
    )

    parser.add_argument(
        "--client_num_per_round",
        type=int,
        default=10,
        metavar="NN",
        help="number of workers",
    )

    parser.add_argument("--gpu_worker_num", type=int, default=8, help="total gpu num")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 16)",
    )

    parser.add_argument(
        "--client_optimizer", type=str, default="sgd", help="SGD with momentum; adam"
    )

    parser.add_argument(
        "--backend", type=str, default="MPI", help="Backend for Server and Client"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1,
        metavar="LR",
        help="learning rate (default: 0.0001)",
    )

    parser.add_argument(
        "--wd", help="weight decay parameter;", type=float, default=0.001
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
        metavar="EP",
        help="how many epochs will be trained locally",
    )

    parser.add_argument(
        "--fl_algorithm",
        type=str,
        default="FedAvgSeq",
        help="Algorithm list: FedAvg; FedOPT; FedProx; FedAvgSeq ",
    )

    parser.add_argument(
        "--comm_round",
        type=int,
        default=30,
        help="how many round of communications we should use",
    )

    parser.add_argument(
        "--mu",
        type=float,
        default=1,
        metavar="mu",
        help="variable for FedProx",
    )

    parser.add_argument(
        "--is_mobile",
        type=int,
        default=0,
        help="whether the program is running on the FedML-Mobile server side",
    )

    parser.add_argument(
        "--frequency_of_the_test",
        type=int,
        default=2,
        help="the frequency of the algorithms",
    )

    parser.add_argument(
        "--process_method",
        type=str,
        default="raw",
        help="the feature process method",
    )

    parser.add_argument(
        "--feature_type",
        type=str,
        default="mel_spec",
        help="the feature type",
    )

    parser.add_argument("--gpu_server_num", type=int, default=1, help="gpu_server_num")

    parser.add_argument(
        "--gpu_num_per_server", type=int, default=4, help="gpu_num_per_server"
    )

    parser.add_argument("--starting_gpu", type=int, default=0)

    parser.add_argument(
        "--gpu_mapping_file",
        type=str,
        default=None,
        help="the gpu utilization file for servers and clients. If there is no \
                        gpu_util_file, gpu will not be used.",
    )

    parser.add_argument(
        "--gpu_mapping_key",
        type=str,
        default="mapping_default",
        help="the key in gpu utilization file",
    )

    parser.add_argument(
        "--grpc_ipconfig_path",
        type=str,
        default="grpc_ipconfig.csv",
        help="config table containing ipv4 address of grpc server",
    )

    parser.add_argument(
        '--test_fold', type=int, default=1, help='Test fold id for Crema-D dataset, default test fold is 1'
    )

    parser.add_argument('--fl_feature', type=bool, default=False,
                        help='raw data or nosiy data')

    parser.add_argument('--label_nosiy', type=bool, default=False,
                        help='clean label or nosiy label')

    parser.add_argument('--label_nosiy_level', type=float, default=0.5,
                        help='nosiy level for labels; 0.9 means 90% wrong')

    parser.add_argument('--db_level', type=float, default=10,
                        help='snr level for the audio (20,30,40)')

    parser.add_argument("--ci", type=int, default=0, help="CI")

    parser.add_argument('--server_optimizer', type=str, default='adam',
                        help='server_optimizer')

    parser.add_argument('--server_lr', type=float, default=0.001,
                        help='server_lr')
    
    parser.add_argument(
        "--setup",
        type=str,
        default="federated",
        help="setup of the experiment: centralized/federated",
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="alpha in direchlet distribution",
    )

    args = parser.parse_args()
    return args


def validate_args(args):
    """
    args : args for Fed Speech
    """
    # Check feature match or not
    if args.process_method not in global_constant.audio_feat_dim_dict:
        raise Exception("Process method not found for " + args.dataset)
    if (
        args.feature_type
        not in global_constant.audio_feat_dim_dict[args.process_method]
    ):
        raise Exception(
            "Feature type not found for "
            + args.dataset
            + " using "
            + args.process_method
        )

    # Training settings possible or not
    # if args.dataset == "iemocap":
        # if int(args.client_num_in_total) != 8:
        #     raise Exception(
        #         "Total number of clients does not match with " + args.dataset
        #     )


def load_data(args, dataset_name):
    if dataset_name == "gcommand":
        if args.fl_feature:
            save_file_name = (
                "speech_commands/federated_dataset_"
                + args.process_method
                + "_"
                + args.feature_type
                + "_db" 
                + str(args.db_level)
                + ".p"
            )
            logging.info("Processing the nosiy data with snr level %s" % str(args.db_level))
        else:
            save_file_name = (
                "speech_commands/federated_dataset_"
                + args.process_method
                + "_"
                + args.feature_type
                + ".p"
            )
            logging.info('Processing the raw data')
        load_file_path = args.data_dir + save_file_name
        dataset = pickle.load(open(load_file_path, "rb"))
        logging.info("dataset has been loaded from saved file")
    if dataset_name == 'crema_d':
        if args.fl_feature:
            save_file_name = (
                "crema_d/federated_dataset_"
                + args.process_method
                + "_"
                + args.feature_type
                + "_fold_"
                + str(args.test_fold)
                +"_db"
                + str(args.db_level)
                + ".p"
            )
            logging.info("Processing the nosiy data with snr level %s" % str(args.db_level))
        else:
            save_file_name = (
                "crema_d/federated_dataset_"
                + args.process_method
                + "_"
                + args.feature_type
                + "_fold_"
                + str(args.test_fold)
                + ".p"
            )
            logging.info('Processing the raw data')
        load_file_path = args.data_dir + save_file_name
        dataset = pickle.load(open(load_file_path, "rb"))
        logging.info("dataset has been loaded from saved file")
    if dataset_name == 'urban_sound':
        if args.fl_feature:
            save_file_name = (
                "urban_sound/federated_dataset_"
                + args.process_method
                + "_"
                + args.feature_type
                + "_fold_"
                + str(args.test_fold)
                +"_alpha"+str(args.alpha).replace(".", "")
                +"_db"+str(args.db_level)
                + ".p"
            )
            load_file_path = args.data_dir + save_file_name
            dataset = pickle.load(open(load_file_path, "rb"))
            logging.info("dataset has been loaded from saved file")
            logging.info("Processing the nosiy data with snr level %s" % save_file_name)
        else:
            save_file_name = (
                "urban_sound/federated_dataset_"
                + args.process_method
                + "_"
                + args.feature_type
                + "_fold_"
                + str(args.test_fold)
                +"_alpha"+str(args.alpha).replace(".", "")
                + ".p"
            )
            load_file_path = args.data_dir + save_file_name
            dataset = pickle.load(open(load_file_path, "rb"))
            logging.info("dataset has been loaded from saved file")
    elif dataset_name == "iemocap":
        if args.fl_feature:
            save_file_name = (
                "iemocap/federated_dataset_"
                + args.process_method
                + "_"
                + args.feature_type
                + "_Session"
                + str(args.test_fold)
                +"_db"
                + str(args.db_level)
                + ".p"
            )
            logging.info("Processing the nosiy data with snr level %s" % str(args.db_level))
        else:
            save_file_name = (
                "iemocap/federated_dataset_"
                + args.process_method
                + "_"
                + args.feature_type
                + "_Session"
                + str(args.test_fold)
                + ".p"
            )
            logging.info('Processing the raw data')
        load_file_path = args.data_dir + save_file_name
        dataset = pickle.load(open(load_file_path, "rb"))
        logging.info("dataset has been loaded from saved file")
        args.client_num_per_round = len(dataset[4])
    elif dataset_name == "meld":
        save_file_name = (
                "meld/federated_dataset_"
                + args.process_method
                + "_"
                + args.feature_type
                + ".p"
        )
        load_file_path = args.data_dir + save_file_name
        dataset = pickle.load(open(load_file_path, "rb"))
        logging.info("dataset has been loaded from saved file")
    return dataset

# def label_nosiy(args, train_data_local_dict, class_num):
#     mean = 0
#     mu = args.label_nosiy_level
#     nosiy_list = np.random.normal(mean, mu, len(train_data_local_dict))
#     nosiy_list = np.absolute(nosiy_list)
#     nosiy_list[nosiy_list > 1.0] = 1.0
#     count = 0
#     for key, data in enumerate(tqdm(train_data_local_dict)):
#         tmp_dataset = []
#         original_data = train_data_local_dict[key].dataset
#         nosiy_level = nosiy_list[count]
#         for i in range(len(original_data)):
#             tmp_dataset_cell = [0, 0]
#             # add label nosiy
#             orginal_label = original_data[i][1].numpy()
#             p1 = nosiy_level/(class_num-1)*np.ones(class_num)
#             p1[orginal_label] = 1-nosiy_level
#             new_label = np.random.choice(class_num,p=p1)
#             tmp_dataset_cell.append(new_label)
#             original_raw_data = original_data[i][0].numpy()
#             tmp_dataset_cell.append(original_raw_data)
#             tmp_dataset.append(tmp_dataset_cell)
#         train_dataset = DatasetGenerator(tmp_dataset)
#         train_data_local_dict[key] = DataLoader(
#             dataset=train_dataset,
#             batch_size=args.batch_size,
#             shuffle=True,
#             collate_fn=collate_fn_padd,
#         )
#         count = count + 1
#     return train_data_local_dict

def label_nosiy(args, train_data_local_dict, class_num):
    for key, data in enumerate(tqdm(train_data_local_dict)):
        #create matrix for each user
        noisy_level = args.label_nosiy_level
        sparse_level = 0.4
        prob_matrix = [1-noisy_level] * class_num * class_num
        sparse_elements = np.random.choice(class_num*class_num, round(class_num*(class_num-1)*sparse_level))
        for idx in range(len(sparse_elements)):
            while sparse_elements[idx]%(class_num+1) == 0:
                sparse_elements[idx] = np.random.choice(class_num*class_num, 1)
            prob_matrix[sparse_elements[idx]] = 0

        available_spots = np.argwhere(np.array(prob_matrix) == 1 - noisy_level)
        for idx in range(class_num):
            available_spots = np.delete(available_spots, np.argwhere(available_spots == idx*(class_num+1)))

        # if key == 12:
        #     a = np.reshape(prob_matrix, (class_num, class_num))
        #     logging.info('prob_matrix={}'.format(a))
        #     logging.info('available={}'.format(available_spots))

        for idx in range(class_num):
            row = prob_matrix[idx*4:(idx*4)+4]
            if len(np.where(np.array(row) == 1 - noisy_level)[0]) == 2:
                unsafe_points = np.where(np.array(row) == 1 - noisy_level)[0]
                unsafe_points = np.delete(unsafe_points, np.where(np.array(unsafe_points) == idx*(class_num+1))[0])
                available_spots = np.delete(available_spots, np.argwhere(available_spots == unsafe_points[0]))
            if np.sum(row) == 1 - noisy_level:
                # logging.info('row = {}'.format(row))
                # logging.info('prob_matrix = {}'.format(prob_matrix))
                zero_spots = np.where(np.array(row) == 0)[0]
                # logging.info('zero spot = {}'.format(zero_spots))
                # logging.info('before prob_matrix[zero_spots[0] + idx * 4]={}'.format(prob_matrix[zero_spots[0] + idx * 4]))
                # logging.info('before prob_matrix[available_spots[0]]={}'.format(prob_matrix[available_spots[0]]))
                prob_matrix[zero_spots[0] + idx * 4], prob_matrix[available_spots[0]] = prob_matrix[available_spots[0]], prob_matrix[zero_spots[0] + idx * 4]
                # logging.info('after prob_matrix[zero_spots[0] + idx * 4]={}'.format(prob_matrix[zero_spots[0] + idx * 4]))
                # logging.info('after prob_matrix[available_spots[0]]={}'.format(prob_matrix[available_spots[0]]))
                available_spots = np.delete(available_spots, 0) 

        prob_matrix = np.reshape(prob_matrix, (class_num, class_num))
        # if key == 12:
        #     logging.info('prob_matrix={}'.format(prob_matrix))

        for idx in range(len(prob_matrix)):
            zeros = np.count_nonzero(prob_matrix[idx]==0)
            if class_num-zeros-1 == 0:
                prob_element = 0
            else:
                prob_element = (noisy_level) / (class_num-zeros-1)
            prob_matrix[idx] = np.where(prob_matrix[idx] == 1-noisy_level, prob_element, prob_matrix[idx])
            prob_matrix[idx][idx] = 1-noisy_level

        # if key == 12:
        #     logging.info('prob_matrix={}'.format(prob_matrix))

        tmp_dataset = []
        original_data = train_data_local_dict[key].dataset
        for i in range(len(original_data)):
            tmp_dataset_cell = [0, 0]
            # add label nosiy
            orginal_label = original_data[i][1].numpy()
            # if key == 12:
            #     logging.info('prob_matrix[orginal_label]={}'.format(prob_matrix[orginal_label]))
            new_label = np.random.choice(class_num,p=prob_matrix[orginal_label])
            tmp_dataset_cell.append(new_label)
            original_raw_data = original_data[i][0].numpy()
            tmp_dataset_cell.append(original_raw_data)
            tmp_dataset.append(tmp_dataset_cell)
        train_dataset = DatasetGenerator(tmp_dataset)
        train_data_local_dict[key] = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn_padd,
        )
    return train_data_local_dict

def create_model(args):
    model = None
    if args.model.startswith("VGG"):
        model = VGG(args.model)
    if args.model == "LeNet":
        model = LeNet()
    if args.model == "audio_conv_rnn":
        feature_size = global_constant.audio_feat_dim_dict[args.process_method][
            args.feature_type
        ]
        label_size = global_constant.label_dim_dict[args.dataset]
        model = audio_conv_rnn(
            feature_size=feature_size, dropout=0.1, label_size=label_size
        )
    elif args.model == "audio_rnn":
        feature_size = global_constant.audio_feat_dim_dict[args.process_method][
            args.feature_type
        ]
        label_size = global_constant.label_dim_dict[args.dataset]
        model = audio_rnn(feature_size=feature_size, dropout=0.1, label_size=label_size)
    elif args.model == "BC_ResNet":
        model = BCResNet()
    return model


def custom_model_trainer(args, model):
    if args.fl_algorithm == "FedProx":
        return FedProxModelTrainer(model)
    else:
        return MyModelTrainer(model)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_fl_algorithm_initializer(alg_name):
    if alg_name == "FedAvg":
        fl_algorithm = FedML_FedAvgSeq_distributed
    elif alg_name == "FedAvgSeq" or alg_name == "FedProx":
        fl_algorithm = FedML_FedAvgSeq_distributed
    elif alg_name == "FedOPT":
        fl_algorithm = FedML_FedOptSeq_distributed
    # elif alg_name == "FedProx":
    #     fl_algorithm = FedML_FedProx_distributed
    else:
        raise Exception("please do sanity check for this algorithm.")

    return fl_algorithm


if __name__ == "__main__":
    # quick fix for issue in MacOS environment: https://github.com/openai/spinningup/issues/16
    if sys.platform == "darwin":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    args = add_args(argparse.ArgumentParser(description="FedSpeech-Distributed"))
    validate_args(args)
    logger.info(args)

    if process_id == 0:
        if args.dataset == "gcommand":
            test_fold = ""
        else:
            test_fold = "-fd" + str(args.test_fold)
        if args.dataset == "urban_sound":
            alpha_str = "-alpha" + str(args.alpha).replace(".", "")
        else:
            alpha_str = ""
        
        if args.label_nosiy:
            label_noise_str = "-ln" + str(args.label_nosiy_level).replace(".", "")
        else:
            label_noise_str = ""
        
        args.setting_str = str(args.fl_algorithm) + "-r" + str(args.comm_round)
        args.setting_str += "-c" + str(args.client_num_per_round) + "-e" + str(args.epochs)
        args.setting_str += "-lr" + str(args.lr) + "-bs" + str(args.batch_size)
        args.setting_str += "-" + args.model + "-" + args.dataset
        args.setting_str += "-" + args.process_method + "-" + args.feature_type
        args.setting_str += test_fold + alpha_str + label_noise_str
        
        result_path = Path.cwd().joinpath("results", "federated", args.dataset)
        args.csv_result_path = str(result_path.joinpath(args.setting_str+".csv"))
        Path.mkdir(Path(result_path), parents=True, exist_ok=True)
        args.best_metric = 0
        
        wandb.init(
            # mode="disabled",
            project="fedaudio",
            entity="ultrazt",
            name=str(args.fl_algorithm)
            + "-r"
            + str(args.comm_round)
            + "-c"
            + str(args.client_num_per_round)
            + "-e"
            + str(args.epochs)
            + "-lr"
            + str(args.lr)
            + "-bs"
            + str(args.batch_size)
            + "-"
            + args.model
            + "-"
            + args.dataset
            + "-"
            + args.process_method
            + "-"
            + args.feature_type
            + test_fold
            + alpha_str
            + label_noise_str,
            config=args,
        )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    set_seed(0)

    # Please check "GPU_MAPPING.md" to see how to define the topology
    logging.info("process_id = %d, size = %d" % (process_id, worker_number))

    # server_device_index = args.starting_gpu
    if process_id == 0:
        device = torch.device(
            "cuda:" + str(args.starting_gpu) if torch.cuda.is_available() else "cpu"
        )
    else:
        process_gpu_dict = dict()
        for client_index in range(args.gpu_worker_num):
            gpu_index = client_index % args.gpu_num_per_server + args.starting_gpu
            process_gpu_dict[client_index] = gpu_index

        logging.info(process_gpu_dict)
        device = torch.device(
            "cuda:" + str(process_gpu_dict[process_id - 1])
            if torch.cuda.is_available()
            else "cpu"
        )
    args.device = device
    logger.info(device)

    # load data and model
    dataset = load_data(args, args.dataset)
    [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ] = dataset

    # fix client number by naturally niid
    args.client_num_in_total = len(train_data_local_num_dict)

    # label nosiy or not
    if args.label_nosiy:
        train_data_local_dict = label_nosiy(args, train_data_local_dict, class_num)
    
    # load model and trainer
    model = create_model(args)
    model_trainer = custom_model_trainer(args, model)
    logging.info(model)

    # start "federated averaging (FedAvg)"
    fl_alg = get_fl_algorithm_initializer(args.fl_algorithm)
    fl_alg(
        process_id,
        worker_number,
        device,
        comm,
        model,
        train_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        args,
        model_trainer,
    )
