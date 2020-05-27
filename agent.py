import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from model import ActorCriticNetwork

class ActorCriticAgent():
    def __init__(self, alpha, inputChannels, gamma=0.99, numActions=12):
        self.gamma = gamma
        self.actorCritic = ActorCriticNetwork(
            alpha=alpha, 
            inputChannels=inputChannels, 
            numActions=numActions)
        self.logProbs = None

    def chooseAction(self, observation):
        policy, _ = self.actorCritic.forward(observation)
        policy = F.softmax(policy, dim=0)
        actionProbs = torch.distributions.Categorical(policy)
        action = actionProbs.sample()
        self.logProbs = actionProbs.log_prob(action)
        return action.item()

    def learn(self, state, reward, nextState, done):
        self.actorCritic.optimizer.zero_grad()

        _, criticValue = self.actorCritic.forward(state)
        _, nextCriticValue = self.actorCritic.forward(nextState)

        reward = torch.tensor(reward, dtype=torch.float).to(self.actorCritic.device)
        delta = reward + self.gamma * nextCriticValue * (1 - int(done)) - criticValue

        actorLoss = -self.logProbs * delta
        criticLoss = delta**2

        (actorLoss + criticLoss).backward()
        self.actorCritic.optimizer.step()

if __name__ == "__main__":
    inputShape = (3, 240, 256)

    #   make agent
    agent = ActorCriticAgent(alpha=0.0001, inputChannels=3)

    #   make fake env observation
    x = torch.ones(inputShape).to(agent.actorCritic.device)
    x = x.unsqueeze(0)
    print(x.shape)
    
    #   compute some actions
    actionChoice = agent.chooseAction(x)
    print(actionChoice)

    #   test learn
    agent.learn(x, 100.0, x, False)
