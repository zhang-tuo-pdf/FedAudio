#!/usr/bin/env bash
DATA_SET=$1
PROCESS_METHOD=$2
FEATURE_TYPE=$3
SAMPLE_NUM=$4
WORKER_NUM=$5
ROUND=$6
EPOCH=$7
BATCH_SIZE=$8
LR=$9
STARTING_GPU_INDEX=${10}
GPU_NUM_PER_SERVER=${11}

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

# sh run_distributed.sh gcommand raw mel_spec 100 8 1000 1 16 0.1 0 8

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./distributed_main.py \
  --dataset $DATA_SET \
  --process_method $PROCESS_METHOD \
  --feature_type $FEATURE_TYPE \
  --client_num_per_round $SAMPLE_NUM \
  --gpu_worker_num $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --starting_gpu $STARTING_GPU_INDEX \
  --gpu_num_per_server $GPU_NUM_PER_SERVER