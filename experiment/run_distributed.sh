#!/usr/bin/env bash
DATA_SET=$1
CLIENT_NUM=$2
SAMPLE_NUM=$3
WORKER_NUM=$4
ROUND=$5
EPOCH=$6
BATCH_SIZE=$7
LR=$8
STARTING_GPU_INDEX=$9
GPU_NUM_PER_SERVER=${10}

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

# sh run_distributed.sh 2118 10 8 30 1 16 0.1 0 8

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./distributed_main.py \
  --dataset $DATA_SET \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $SAMPLE_NUM \
  --gpu_worker_num $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --starting_gpu $STARTING_GPU_INDEX \
  --gpu_num_per_server $GPU_NUM_PER_SERVER