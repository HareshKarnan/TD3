import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
import random

# files = glob.glob('logs/lambda_expts/*')
# files = glob.glob('logs/model_free/*')
# files = glob.glob('logs/fwd_inv_model/*')
# files = glob.glob('logs/fwd_model/*')


def extract_expt_data(files):
    data = {}
    for fileNamefull in files:
        fileName = fileNamefull.split('/')[-1]
        # print(fileName)
        totalTimesteps = fileName.split('_')[6]
        modelIters = fileName.split('_')[9]
        modelGrads = fileName.split('_')[12]
        stateNoise = fileName.split('_')[14]
        seed = fileName.split('_')[15]

        # print(modelIters, modelGrads, seed)
        # first branch of dict -> model iterations
        if modelIters not in data.keys(): data[modelIters] = {}

        # second branch of dict -> model gradient steps
        if modelGrads not in data[modelIters].keys(): data[modelIters][modelGrads] = {}

        # third branch of dict -> model gradient steps -> modelgrads
        if stateNoise not in data[modelIters][modelGrads].keys(): data[modelIters][modelGrads][stateNoise] = {}

        # third branch of dict -> experiment seed
        if seed not in data[modelIters][modelGrads][stateNoise].keys(): data[modelIters][modelGrads][stateNoise][seed] = {}

        # store the data of all seeds here
        with open(fileNamefull+'/log.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            episode, episode_rewards = [], []
            curr_episode = 0
            for row in csvreader:
                reward, done, episode_num, episode_reward, episode_timesteps, total_timesteps = row
                if done == 'True':
                    episode.append(int(episode_num))
                    episode_rewards.append(float(episode_reward))

        data[modelIters][modelGrads][stateNoise][seed]['episode_rewards'] = episode_rewards
        data[modelIters][modelGrads][stateNoise][seed]['episode'] = episode

    return data

data = extract_expt_data(glob.glob('logs/dual_state_expl_2/*'))

# find the mean and std across expts
ax = plt.subplot()

for modelIters in data.keys():
    for modelGrads in data[modelIters].keys():
        for stateNoise in data[modelIters][modelGrads].keys():
            color = (random.random(), random.random(), random.random())
            # print('found seeds :: ', data[modelIters][modelGrads].keys())
            dataX, dataY = [], []

            for seed in data[modelIters][modelGrads][stateNoise].keys():
                episode = data[modelIters][modelGrads][stateNoise][seed]['episode']
                episodeRew = data[modelIters][modelGrads][stateNoise][seed]['episode_rewards']
                dataY.append(episodeRew)
                dataX.append(episode)

            minX = min([val[-1] for val in dataX]) + 1

            dataX = [datX[:minX] for datX in dataX]
            dataY = [datY[:minX] for datY in dataY]
            dataY = np.asarray(dataY)

            ax.plot(dataX[0], np.mean(dataY, axis=0), color=color, label='MI_'+modelIters+'_MG_'+modelGrads+'_SN_'+stateNoise)
            ax.fill_between( dataX[0],
                             np.mean(dataY, axis=0) - np.std(dataY, axis=0)/np.sqrt(10),
                             np.mean(dataY, axis=0) + np.std(dataY, axis=0)/np.sqrt(10),
                             alpha=0.25,
                              color=color)

            plt.legend()

plt.xlabel('Episodes')
plt.ylabel('Episode Returns')
plt.show()

