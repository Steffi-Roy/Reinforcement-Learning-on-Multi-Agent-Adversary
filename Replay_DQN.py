##Implement Replay Buffer and DQN network Arch


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from collections import namedtuple, deque

"""Transition is a namedtuple used to store a transition.

The structure of Transition looks like this:
    (state, action, reward, next_state, done)
"""
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:

    def __init__(self, capacity=100_000):

        self._storage = deque([], maxlen=capacity)

    def add(self, state, action, reward, next_state, done):

        transition = Transition(state, action, reward, next_state, done)
        self._storage.append(transition)

    def sample(self, batch_size):

        transitions = random.sample(self._storage, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        """Return the length of the buffer."""
        return len(self._storage)


class SimpleNet(nn.Module):
    """SimpleNet is a 3-layer simple neural network.
    It's used to approximate the policy functions and the value functions.
    """

    def __init__(self, input_dim, output_dim, hidden_dim):

        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


