import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from agent import ActorCriticAgent

from collections import deque

import gym
import math
from matplotlib import pyplot as plt

'''
-tensorboard
-checkpointing
-worker frame collecting
-batch ppo
'''



if __name__ == '__main__':
    
    #   make agent
    # inputShape = (10, 240, 256)
    agent = ActorCriticAgent(alpha=0.001, inputChannels=1, gamma=0.99, numActions=7)

    #   make env
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    env = gym_super_mario_bros.make('SuperMarioBros-v1')
    # env = gym_super_mario_bros.make('SuperMarioBros-2-1-v1')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    scoreHistory = []
    numHiddenEpisodes = -1
    highScore = -math.inf
    recordTimeSteps = math.inf

    aTimerDuration = 20
    aTimer = 0

    episode = 0
    while True:
        stuckTimer = 20
        x_pos_screen_buffer = deque(maxlen=stuckTimer)
        stuck = False
        fixingStuck = False
        
        frameBufferSize = 9
        frameBuffer = deque(maxlen=frameBufferSize)
        
        done = False
        state = env.reset()

        #   prep observation
        state = np.rollaxis(state, 2, 0)
        state = state.mean(axis=0)
        # print(state)
        # print("rgb mix")
        # print(state.shape)

        for i in range(frameBufferSize):
            frameBuffer.append(state)

        tempFrameBuffer = list(frameBuffer)[::-3]
        # print(len(tempFrameBuffer))
        # now, now + 3, now + 6
        observation = np.stack(frameBuffer)
        observation = observation.mean(axis=0)
        observation = observation.reshape(1, *observation.shape)

        observation = torch.tensor(observation.copy()).float()
        observation = observation.to(agent.actorCritic.device)
        observation = observation.unsqueeze(0)

        # print("gpu ready obs tensor")
        # print(observation.shape)

        score, frame = 0, 1
        while not done:
            if episode > numHiddenEpisodes:
                env.render()


            action = agent.chooseAction(observation)

            if fixingStuck:
                aTimer -= 1
                action = 2
                if aTimer == 0:
                    fixingStuck = False
                    stuck = False
                    action = 0
            
            state_, reward, done, info = env.step(action)

            if fixingStuck:    
                reward = 0

            #   reward shaping :/
            # y_pos = info['y_pos']
            # reward += y_pos / 5.0

            #   check for stuckness
            x_pos_screen = info['x_pos']
            x_pos_screen_buffer.append(x_pos_screen)
            if len(x_pos_screen_buffer) == stuckTimer:
                # print(x_pos_screen_buffer)
                last_x_post = x_pos_screen_buffer[-1]
                stuck = all( x_pos == last_x_post for x_pos in x_pos_screen_buffer)
                if not fixingStuck:
                    if stuck:
                        fixingStuck = True
                        aTimer = aTimerDuration
            

            #   prep nextObservation
            state_ = np.rollaxis(state_, 2, 0)
            state_ = state_.mean(axis=0)

            frameBuffer.append(state_)

            # if frame == 100:
            #     exampleFrame = observation.cpu().numpy()
            #     exampleFrame = exampleFrame.reshape(*exampleFrame.shape[2:])
            #     plt.imshow(exampleFrame)
            #     plt.show()

            tempFrameBuffer = list(frameBuffer)[::-3]
            nextObservation = np.stack(tempFrameBuffer)
            nextObservation = nextObservation.mean(axis=0)
            nextObservation = nextObservation.reshape(1, *nextObservation.shape)

            nextObservation = torch.tensor(nextObservation.copy()).float()
            nextObservation = nextObservation.to(agent.actorCritic.device)
            nextObservation = nextObservation.unsqueeze(0)

            # if frame == 100:
            #     exampleFrame = nextObservation.cpu().numpy()
            #     exampleFrame = exampleFrame.reshape(*exampleFrame.shape[2:])
            #     plt.imshow(exampleFrame)
            #     plt.show()

            agent.learn(observation, reward, nextObservation, done)
            observation = nextObservation
            score += reward
            frame += 1

            # if stuck:
            #     break

        scoreHistory.append(score)

        recordTimeSteps = min(recordTimeSteps, frame)
        highScore = max(highScore, score)
        print(( "ep {}: high-score {:12.3f}, shortest-time {:d}, "
                "score {:12.3f}, last-epidode-time {:4d}").format(
            episode, 
            highScore, 
            recordTimeSteps, 
            score,
            frame,
            ))
        episode += 1

    fig = plt.figure()
    meanWindow = 10
    meanedScoreHistory = np.convolve(scoreHistory, np.ones(meanWindow), 'valid') / meanWindow
    plt.plot(np.arange(0, numEpisodes-1, 1.0), meanedScoreHistory)    
    plt.ylabel("score")
    plt.xlabel("episode")
    plt.title("Training Scores")
    plt.show()