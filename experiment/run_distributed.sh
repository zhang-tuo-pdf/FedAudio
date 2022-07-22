#!/usr/bin/env bash

CLIENT_NUM=$1
SAMPLE_NUM=$2
WORKER_NUM=$3
ROUND=$4
EPOCH=$5
BATCH_SIZE=$6
LR=$7
STARTING_GPU_INDEX=$8
GPU_NUM_PER_SERVER=$9

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

#sh run_distributed.sh 2118 10 8 30 1 16 0.1 0 8

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./distributed_main.py \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $SAMPLE_NUM \
  --gpu_worker_num $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --starting_gpu $STARTING_GPU_INDEX \
  --gpu_num_per_server $GPU_NUM_PER_SERVER