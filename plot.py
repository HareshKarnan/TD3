import matplotlib.pyplot as plt
import csv
import numpy as np

file = 'logs/TD3_env_Hopper-v2_seed_0_MB_forward_timesteps_1000000_M_iters_10_M_grad_10.csv'

with open(file, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    episode, episode_rewards = [], []
    curr_episode = 0
    for row in csvreader:
        reward, done, episode_num, episode_reward, episode_timesteps, total_timesteps = row
        if done == 'True':
            episode.append(int(episode_num))
            episode_rewards.append(float(episode_reward))

assert len(episode)==len(episode_rewards), "x and y data is not same in length"

plt.plot(episode, episode_rewards, '-b')
plt.xlabel('Episodes')
plt.ylabel('Episode Return')
plt.show()