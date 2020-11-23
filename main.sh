#!/bin/bash


source /u/haresh92/.bashrc
source /scratch/cluster/haresh92/venv/bin/activate

which python

python3.6 main.py \
  --seed $1 \
  --max_timesteps $2 \
  --policy_name "TD3" \
  --fwd_model_update_freq $3 \
  --env_name $4 \
  --model_based forward \
  --log_training \
  --model_iters $5 \
  --log_path "/u/pstone/agents6/haresh/MBTD3/experiment/" \
  --model_gradient_times $6 &

wait
