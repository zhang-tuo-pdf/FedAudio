#!/usr/bin/env bash
for i in {4..4}
  do
  mpirun -np 3 -hostfile ./mpi_host_file python3 ./distributed_main.py \
    --dataset iemocap \
    --process_method pretrain \
    --feature_type apc \
    --client_num_per_round 8 \
    --gpu_worker_num 2 \
    --comm_round 300 \
    --epochs 1 \
    --batch_size 16 \
    --lr 0.005 \
    --starting_gpu 0 \
    --gpu_num_per_server 2 \
    --test_fold $i
done

  # gcommand
  # sh run_distributed.sh gcommand raw mel_spec 106 8 5000 1 16 0.1 0 8
  # iemocap
  # sh run_distributed.sh iemocap pretrain apc 8 8 5000 1 16 0.0025 0 8
  # crema-d
  # sh run_distributed.sh crema_d pretrain apc 7 7 1000 1 16 0.1 0 7
  # urban_sound
  # sh run_distributed.sh urban_sound raw mel_spec 10 8 5000 1 16 0.1 0 8