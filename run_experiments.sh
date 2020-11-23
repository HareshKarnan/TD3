#!/bin/bash

# Script to reproduce results
export CUDA_VISIBLE_DEVICES=0

for((i=0;i<1;i+=1))
do 

  # model free
  python3.6 main.py \
  --max_timesteps 1000000 \
  --policy_name "TD3" \
  --env_name "InvertedPendulum-v2" \
  --log_training \
  --seed $i &

#	# backward model based
#	python main.py \
#	--save_models \
#  --max_timesteps 1000000 \
#  --policy_name "TD3" \
#  --bwd_model_update_freq 5e3 \
#  --env_name "Hopper-v2" \
#  --seed $i \
#  --model_based backward \
#  --model_iters 10 &

  # forward model based
#  python3.6 main.py \
#	--save_models \
#  --max_timesteps 6000 \
#  --policy_name "TD3" \
#  --fwd_model_update_freq 5e3 \
#  --env_name "InvertedPendulum-v2" \
#  --seed $i \
#  --model_based forward \
#  --log_training \
#  --model_iters 2 &

done

wait