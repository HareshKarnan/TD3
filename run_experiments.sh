#!/bin/bash

# Script to reproduce results
export CUDA_VISIBLE_DEVICES=0

for((i=0;i<1;i+=1))
do

  # forward model based
  python3.6 main.py \
  --max_timesteps 1000000 \
  --policy_name "TD3" \
  --fwd_model_update_freq 5e3 \
  --env_name "InvertedPendulumPyBulletEnv-v0" \
  --seed $i \
  --model_based "dual" \
  --log_training \
  --model_gradient_times 2 \
  --model_iters 5 &

done

wait