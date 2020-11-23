#!/bin/bash

source ~/.bashrc
#conda activate myenv

rm /u/pstone/agents6/haresh/MBTD3/logs/*
echo "Cleared /u/pstone/agents6/haresh/MBTD3/logs/*"

# set which GPU to use
export CUDA_VISIBLE_DEVICES=0

# number of seeds to run
num_seeds=1

which python

time_steps=(1000000)
fwd_update_freq=(5e3)
envs=("InvertedPendulum-v2")
model_iters=(2 10 100)
model_gradient_times=(1 2 5 10)
n=1

# script to run experiments
for ((i=0;i<$num_seeds;i++)) # for running multiple seeds
do
  for timest in "${time_steps[@]}" # for running multiple clip ranges
  do
    for fwd_freq in "${fwd_update_freq[@]}" # for running multiple clip ranges
    do
      for env in "${envs[@]}"
      do
        for model_iter_num in "${model_iters[@]}"
        do
          for model_gradnum in "${model_gradient_times[@]}"
          do
            python3.6 condorizer.py -o '/u/pstone/agents6/haresh/MBTD3/condor_log/'$n\
            bash main.sh  \
            $i \
            $timest \
            $fwd_freq \
            $env \
            $model_iter_num \
            $model_gradnum \

            n=$(($n+1))
            sleep 0.001
          done
        done
      done
    done
  done
  wait
  echo " ~~~ Experiment Completed :) ~~"
done