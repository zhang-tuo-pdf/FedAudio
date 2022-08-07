import argparse
import logging
import os
import random
import socket
import sys
import pickle

import numpy as np
import torch
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
        help="input batch size for training (default: 64)",
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
        default=5,
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
        '--test_fold', type=int, default=10, help='Test fold id for Crema-D dataset, default test fold is 1'
    )

    parser.add_argument('--fl_feature', type=bool, default=True,
                        help='raw data or nosiy data')

    parser.add_argument('--db_level', type=float, default=20,
                        help='snr level for the audio (20,30,40)')

    parser.add_argument("--ci", type=int, default=0, help="CI")

    parser.add_argument('--server_optimizer', type=str, default='adam',
                        help='server_optimizer')

    parser.add_argument('--server_lr', type=float, default=0.001,
                        help='server_lr')

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
                "speech_commands/processed_dataset_"
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
                "speech_commands/processed_dataset_"
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
        save_file_name = (
            "crema_d/processed_dataset_"
            + args.process_method
            + "_"
            + args.feature_type
            + "_fold_"
            + str(args.test_fold)
            + ".p"
        )
        load_file_path = args.data_dir + save_file_name
        dataset = pickle.load(open(load_file_path, "rb"))
        logging.info("dataset has been loaded from saved file")
    if dataset_name == 'urban_sound':
        save_file_name = (
            "urban_sound/processed_dataset_"
            + args.process_method
            + "_"
            + args.feature_type
            + "_fold_"
            + str(args.test_fold)
            + ".p"
        )
        load_file_path = args.data_dir + save_file_name
        dataset = pickle.load(open(load_file_path, "rb"))
        logging.info("dataset has been loaded from saved file")
    elif dataset_name == "iemocap":
        save_file_name = (
            "iemocap/processed_dataset_"
            + args.process_method
            + "_"
            + args.feature_type
            + "_Session"
            + str(args.test_fold)
            + ".p"
        )
        load_file_path = args.data_dir + save_file_name
        dataset = pickle.load(open(load_file_path, "rb"))
        logging.info("dataset has been loaded from saved file")
    elif dataset_name == "meld":
        save_file_name = (
                "meld/processed_dataset_"
                + args.process_method
                + "_"
                + args.feature_type
                + ".p"
        )
        load_file_path = args.data_dir + save_file_name
        dataset = pickle.load(open(load_file_path, "rb"))
        logging.info("dataset has been loaded from saved file")
    return dataset


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
        wandb.init(
            # mode="disabled",
            project="fedaudio",
            entity="ultrazt",
            name=str(args.fl_algorithm)
            + "-r"
            + str(args.comm_round)
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
            + args.feature_type,
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
