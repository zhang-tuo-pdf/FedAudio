#!/usr/bin/env bash

CLIENT_NUM=$1
WORKER_NUM=$2
ROUND=$3
EPOCH=$4
BATCH_SIZE=$5
LR=$6
STARTING_GPU_INDEX=$7
GPU_NUM_PER_SERVER=$8

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

#sh run_distributed.sh 2118 10 30 1 16 0.1 0 8

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./distributed_main.py \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --starting_gpu $STARTING_GPU_INDEX \
  --gpu_num_per_server $GPU_NUM_PER_SERVER