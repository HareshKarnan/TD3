#!/bin/bash
# forward model based approach

# Script to reproduce results
export CUDA_VISIBLE_DEVICES=0
num_seeds=1
fwd_update_freq=(5e3)
env_names=("InvertedPendulumPyBulletEnv-v0")
model_iters=(2 4 8 10)
model_gradient_times=(2 4 8 10)


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
          --max_timesteps 1000000 \
          --policy_name "TD3" \
          --fwd_model_update_freq $fwd_freq \
          --env_name $env_name \
          --seed $i \
          --model_based forward \
          --log_training \
          --model_gradient_times $model_grad_time \
          --model_iters $model_iter &

        done
      done
    done
  done
done
wait
