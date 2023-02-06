# based on the original fedavg_seq inside the FedML repo (https://github.com/FedML-AI/FedML/tree/master/python/fedml/simulation/mpi/fedavg_seq)
# and one of the forked version of FedML repo (https://github.com/wizard1203/FedML/tree/master/python/fedml/simulation/mpi/fedavg_seq)
# we add the sequential training on top of the fedavg, which means that each process could represent multiple clients during the training.
# we also change the test function for the F1 score test.


class MyMessage(object):
    """
    message type definition
    """

    # server to client
    MSG_TYPE_S2C_INIT_CONFIG = 1
    MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT = 2

    # client to server
    MSG_TYPE_C2S_SEND_MODEL_TO_SERVER = 3
    MSG_TYPE_C2S_SEND_STATS_TO_SERVER = 4

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_CLIENT_INDEX = "client_idx"

    MSG_ARG_KEY_TRAIN_CORRECT = "train_correct"
    MSG_ARG_KEY_TRAIN_ERROR = "train_error"
    MSG_ARG_KEY_TRAIN_NUM = "train_num_sample"

    MSG_ARG_KEY_TEST_CORRECT = "test_correct"
    MSG_ARG_KEY_TEST_ERROR = "test_error"
    MSG_ARG_KEY_TEST_NUM = "test_num_sample"

    MSG_ARG_KEY_AVG_WEIGHTS = "avg_weights"
    MSG_ARG_KEY_CLIENT_SCHEDULE = "client_schedule"
    MSG_ARG_KEY_CLIENT_RUNTIME_INFO = "client_runtime"
