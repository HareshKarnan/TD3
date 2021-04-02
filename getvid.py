import pybulletgym
import gym
import TD3
import torch
import pickle
import numpy as np
import time

env = gym.make('InvertedPendulumPyBulletEnv-v0')

done = False
env.render(mode="human")
obs = env.reset()
# policy = pickle.load(open('logs/model_free/TD3_env_InvertedPendulumPyBulletEnv-v0_MB_None_timesteps_100000_M_iters_100_M_grad_10_3/policy.pkl','rb'))
# policy = pickle.load(open('logs/dual_final/TD3_env_InvertedPendulumPyBulletEnv-v0_MB_dual_timesteps_200000_M_iters_1_M_grad_2_4/policy.pkl','rb'))
policy = pickle.load(open('logs/fwd_model/TD3_env_InvertedPendulumPyBulletEnv-v0_MB_forward_timesteps_100000_M_iters_1_M_grad_2_9/policy.pkl','rb'))
device = torch.device("cpu")

returnlist = []
times = 0
returnval = 0
while times<3:
    action = policy.actor(torch.FloatTensor(obs).to(device))
    obs, rew, done, _ = env.step(action.detach().numpy())
    returnval += rew

    env.render(mode="human")
    time.sleep(0.05)
    if done:
        obs = env.reset()
        done = False
        times+=1
        returnlist.append(returnval)
        returnval = 0

print('mean :: ', np.mean(returnlist))
print('std :: ', np.std(returnlist))