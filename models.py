import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
import torch.optim as optimizers
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
from tqdm import tqdm


class SimpleMLP(Module):
    def __init__(self, num_kitchens: int, num_items: int, hidden_layers: int = 3, hidden_features: int = 64,
                 data_means: float = 1.0):
        """
        Simple fully connected MLP to assign rescued food to kitchens
        :param num_kitchens: Number of kitchens
        :param num_items: Number of different food categories
        :param hidden_layers: Number of hidden layers
        :param hidden_features: Number of hidden features
        :param data_means: Means of data, obtained from training data and saved for inference
        """
        super(SimpleMLP, self).__init__()

        self.data_means = data_means

        # input consists of inventory for each kitchen plus food item to distribute
        input_dim = (num_kitchens + 1) * num_items

        self.net = []

        # input layer
        self.net.append(nn.Linear(input_dim, hidden_features))
        self.net.append(nn.BatchNorm1d(hidden_features))
        self.net.append(nn.ReLU())

        # hidden layers
        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.BatchNorm1d(hidden_features))
            self.net.append(nn.ReLU())

        # final output layer
        self.net.append(nn.Linear(hidden_features, num_kitchens))
        self.net.append(nn.Softmax(dim=1))

        self.net = nn.Sequential(*self.net)

    def forward(self, x: torch.Tensor, center_data: bool = False) -> torch.Tensor:
        """
        Forward pass. Input is reshaped to [batch_size, (num_kitchens + 1) * num_items]
        :param center_data: If true, data mean will be subtracted before data is passed to network
        :param x: Input with kitchen inventories and food item to distribute
                    dimensions: [batch_size, (num_kitchens + 1), num_items]
        :return: Softmax assignment to kitchens [batch_size, num_kitchens]
        """
        current_batch_size = x.shape[0]
        if center_data:
            x = x - self.data_means
        return self.net(x.reshape(current_batch_size, -1))

