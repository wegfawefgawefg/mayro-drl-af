import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, alpha, inputChannels, numActions):
        super().__init__()
        self.inputChannels = inputChannels
        self.numActions = numActions

        self.maxPool0 = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(self.inputChannels, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.maxPool1 = nn.MaxPool2d(2, 2)

        self.convOutShape = 128 * 5 * 6
        # self.convOutShape = 128 * 13 * 14
        self.fc1Dims = 1024
        self.fc2Dims = 512

        #   primary network
        self.fc1 = nn.Linear(self.convOutShape, self.fc1Dims)
        self.fc2 = nn.Linear(self.fc1Dims, self.fc2Dims)

        #   tail networks
        self.actor = nn.Linear(self.fc2Dims, numActions)
        self.critic = nn.Linear(self.fc2Dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, observation):
        batchSize = observation.shape[0]

        x = self.maxPool0(observation)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxPool1(x)
        x = x.view(batchSize, self.convOutShape)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        policy  = self.actor(x)
        value   = self.critic(x)
        return policy, value

if __name__ == "__main__":
    inputShape = (3, 240, 256)

    net = ActorCriticNetwork(
        alpha=0.001, 
        inputChannels=3, 
        numActions=12)

    #   make fake data
    x = torch.ones(inputShape).to(net.device)
    x = x.unsqueeze(0)
    print(x.shape)

    #   feedforward 
    policy, value = net(x)
    print("policy {}".format(policy))
    print("value {}".format(value))