import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
import random

# files = glob.glob('logs/lambda_expts/*')
# files = glob.glob('logs/model_free/*')
# files = glob.glob('logs/fwd_inv_model/*')
files = glob.glob('logs/fwd_model/*')


def extract_expt_data(files):
    data = {}
    for fileNamefull in files:
        fileName = fileNamefull.split('/')[-1]
        # print(fileName)
        totalTimesteps = fileName.split('_')[6]
        modelIters = fileName.split('_')[9]
        modelGrads = fileName.split('_')[12]
        seed = fileName.split('_')[13]

        # print(modelIters, modelGrads, seed)
        # first branch of dict -> model iterations
        if modelIters not in data.keys(): data[modelIters] = {}

        # second branch of dict -> model gradient steps
        if modelGrads not in data[modelIters].keys(): data[modelIters][modelGrads] = {}

        # third branch of dict -> experiment seed
        if seed not in data[modelIters][modelGrads].keys(): data[modelIters][modelGrads][seed] = {}

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

        data[modelIters][modelGrads][seed]['episode_rewards'] = episode_rewards
        data[modelIters][modelGrads][seed]['episode'] = episode

    return data

fwd_data = extract_expt_data(glob.glob('logs/fwd_model/*'))
fwd_inv_data = extract_expt_data(glob.glob('logs/dual_final/*'))
model_free_data = extract_expt_data(glob.glob('logs/model_free/*'))

# find the mean and std across expts
ax = plt.subplot()

for i, data in enumerate([fwd_data, fwd_inv_data, model_free_data]):
    for modelIters in data.keys():
        for modelGrads in data[modelIters].keys():


            # select the right experiment here
            if i == 0 and (modelIters != '1' or modelGrads != '3'): continue
            if i == 1 and (modelIters != '1' or modelGrads != '2'): continue

            print(i, modelIters, modelGrads, type(modelIters), type(modelGrads))

            color = (random.random(), random.random(), random.random())
            # print('found seeds :: ', data[modelIters][modelGrads].keys())
            dataX, dataY = [], []

            for seed in data[modelIters][modelGrads].keys():
                episode = data[modelIters][modelGrads][seed]['episode']
                episodeRew = data[modelIters][modelGrads][seed]['episode_rewards']
                dataY.append(episodeRew)
                dataX.append(episode)

            minX = min([val[-1] for val in dataX]) + 1

            dataX = [datX[:minX] for datX in dataX]
            dataY = [datY[:minX] for datY in dataY]
            dataY = np.asarray(dataY)
            if i==0:
                label = 'Forward Model'
                color = (0, 0, 1)
                exptnums = 10
                # continue
            elif i==1:
                label = 'Forward + Inverse Model'
                color = (0, 1, 0)
                exptnums = 5

                continue
            elif i==2:
                label = 'Model Free'
                color = (1, 0, 0)
                exptnums = 10

            ax.plot(dataX[0], np.mean(dataY, axis=0), color=color, label=label)
            ax.fill_between( dataX[0],
                             np.mean(dataY, axis=0) - np.std(dataY, axis=0)/np.sqrt(exptnums),
                             np.mean(dataY, axis=0) + np.std(dataY, axis=0)/np.sqrt(exptnums),
                             alpha=0.25,
                              color=color)

            plt.legend()

plt.ylim([0, 900])
plt.xlim([0, 250])
plt.xlabel('Episodes')
plt.ylabel('Episode Returns')
plt.savefig('fwd_model.png')
plt.show()

