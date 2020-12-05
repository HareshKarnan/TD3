#!/bin/bash
# forward model based approach

# Script to reproduce results
export CUDA_VISIBLE_DEVICES=0
num_seeds=5
fwd_update_freq=(5e3)
env_names=("InvertedPendulumPyBulletEnv-v0")
model_iters=(1)
model_gradient_times=(2)
state_expl_noises=(0.005)

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
          for state_exp_noise in "${state_expl_noises[@]}" # for running multiple clip ranges
          do
            python3.6 main.py \
            --max_timesteps 200000 \
            --policy_name "TD3" \
            --fwd_model_update_freq $fwd_freq \
            --env_name $env_name \
            --seed $i \
            --model_based "forward" \
            --log_training \
            --model_gradient_times $model_grad_time \
            --expl_noise 0.005 \
            --state_expl_noise $state_exp_noise \
            --model_iters $model_iter &
          done
        done
      done
    done
  done
done
wait
