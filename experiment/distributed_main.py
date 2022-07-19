import argparse
import logging
import os
import random
import socket
import sys
import pickle

import numpy as np
import psutil
import setproctitle
import torch
import dill
import wandb
from tqdm import tqdm

# add the FedML root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from data_loading.data_loader.gcommand_loader import load_partition_data_audio
from model.vgg_speech import VGG
from model.LeNet import LeNet
from model.bc_resnet import BCResNet
from trainers.speech_trainer import MyModelTrainer
from FedML.fedml_api.distributed.fedavg.FedAvgAPI import (
    FedML_init,
    FedML_FedAvg_distributed,
)
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
        default="BC_ResNet",
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
        "--data_dir", type=str, default="../data/speech_commands", help="data directory"
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
        default="FedAvg",
        help="Algorithm list: FedAvg; FedOPT; FedProx ",
    )

    parser.add_argument(
        "--comm_round",
        type=int,
        default=30,
        help="how many round of communications we shoud use",
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

    parser.add_argument("--ci", type=int, default=0, help="CI")
    args = parser.parse_args()
    return args


def load_data(args, dataset_name):
    # process_method = "pretrain"
    # feature_type = "apc"
    # fl_feature = True
    # snr_level = [20, 30, 40]
    # device_ratio = [round(0.4 * 2118), round(0.3 * 2118), round(0.3 * 2118)]
    if dataset_name == "gcommand":
        load_file_path = "../data/speech_commands/processed_dataset.p"
        dataset = pickle.load(open(load_file_path, "rb"))
    #     data_loader = load_partition_data_audio
    #     (
    #         train_data_num,
    #         test_data_num,
    #         train_data_global,
    #         test_data_global,
    #         train_data_local_num_dict,
    #         train_data_local_dict,
    #         test_data_local_dict,
    #         class_num,
    #     ) = data_loader(
    #         args.batch_size,
    #         process_method,
    #         feature_type=feature_type,
    #         fl_feature=fl_feature,
    #         snr_level=snr_level,
    #         device_ratio=device_ratio,
    #     )
    #
    # dataset = [
    #     train_data_num,
    #     test_data_num,
    #     train_data_global,
    #     test_data_global,
    #     train_data_local_num_dict,
    #     train_data_local_dict,
    #     test_data_local_dict,
    #     class_num,
    # ]
    return dataset


def create_model(args):
    model = None
    if args.model.startswith("VGG"):
        model = VGG(args.model)
    if args.model == "LeNet":
        model = LeNet()
    elif args.model == "BC_ResNet":
        model = BCResNet()
    return model


def custom_model_trainer(args, model):
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
        fl_algorithm = FedML_FedAvg_distributed
    elif alg_name == "FedOPT":
        fl_algorithm = FedML_FedOpt_distributed
    elif alg_name == "FedProx":
        fl_algorithm = FedML_FedProx_distributed
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
    logger.info(args)

    if process_id == 0:
        wandb.init(
            mode="disabled",
            project="fedspeech",
            entity="ultraz",
            name="FedAVG-r"
            + str(args.comm_round)
            + "-e"
            + str(args.epochs)
            + "-lr"
            + str(args.lr)
            + "-bs"
            + str(args.batch_size)
            + "-c"
            + str(args.client_num_in_total)
            + "-"
            + args.model
            + "-"
            + args.dataset,
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
        for client_index in range(args.client_num_in_total):
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
