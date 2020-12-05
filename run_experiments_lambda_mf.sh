#!/bin/bash
# forward model based approach

# Script to reproduce results
export CUDA_VISIBLE_DEVICES=0
num_seeds=10
fwd_update_freq=(5e3)
env_names=("InvertedPendulumPyBulletEnv-v0")
model_iters=(1)
model_gradient_times=(1)


for ((i=0;i<$num_seeds;i+=1))
do
  for fwd_freq in "${fwd_update_freq[@]}" # for running multiple clip ranges
  do
    for env_name in "${env_names[@]}" # for running multiple clip ranges
    do
      for model_iter in "${model_iters[@]}" # for running multiple clip ranges
      do
        for model_grad_time in "${model_gradient_times[@]}" # for running multiple clip ranges
        do
          python3.6 main.py \
          --max_timesteps 100000 \
          --policy_name "TD3" \
          --env_name $env_name \
          --seed $i \
          --log_training &
        done
      done
    done
  done
done
wait
