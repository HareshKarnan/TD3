import pybulletgym
import gym
import time

done = False
env = gym.make('InvertedPendulumPyBulletEnv-v0')
# env = gym.make('InvertedPendulum-v2')
env.render(mode="human")
env.reset()

while not done:
    action = env.action_space.sample()
    obs, rew, done, _ = env.step(action)
    env.render(mode='human', close=False)
    if done:
        done = False
        env.reset()
