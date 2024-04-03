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


class KitchenNet(Module):
    def __init__(self, num_kitchens: int, num_items: int, data_max: torch.Tensor, data_min: torch.Tensor):
        super(KitchenNet, self).__init__()

        self.data_max = data_max
        self.data_min = data_min
        self.data_range = data_max - data_min
        self.data_range[self.data_range == 0] = 1.0

        self.num_kitchens = num_kitchens
        self.num_items = num_items

        self.net = []

    def forward(self, x: torch.Tensor, center_data: bool = False) -> torch.Tensor:
        current_batch_size = x.shape[0]
        if center_data:
            # x = x - self.data_means
            assert self.data_max is not None
            assert self.data_min is not None

            x = (x - self.data_min) / self.data_range
            x = (x - 0.5) * 2.0
        return self.net(x.reshape(current_batch_size, -1))


class SimpleMLP(KitchenNet):
    def __init__(self, num_kitchens: int, num_items: int, hidden_layers: int = 3, hidden_features: int = 64,
                 data_means: float = 0.0, n_components: int = 0,
                 data_max: torch.Tensor = None, data_min: torch.Tensor = None):
        """
        Simple fully connected MLP to assign rescued food to kitchens
        :param num_kitchens: Number of kitchens
        :param num_items: Number of different food categories
        :param hidden_layers: Number of hidden layers
        :param hidden_features: Number of hidden features
        :param data_means: Means of data, obtained from training data and saved for inference
        :param data_max: Maxima of data, obtained from training data and saved for inference
        :param data_min: Minima of data, obtained from training data and saved for inference
        """
        super().__init__(num_kitchens, num_items, data_max, data_min)

        # input consists of inventory for each kitchen plus food item to distribute
        if n_components == 0:
            input_dim = (num_kitchens + 1) * num_items
        else:
            input_dim = n_components

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


class MLPVariableLayers(KitchenNet):
    def __init__(self, num_kitchens: int, num_items: int, hidden_features: list = [64],
                 data_max: torch.Tensor = None, data_min: torch.Tensor = None):
        super().__init__(num_kitchens, num_items, data_max, data_min)

        self.input_dim = (num_kitchens + 1) * num_items
        temp_input_dim = self.input_dim

        for i, hidden_dim in enumerate(hidden_features):
            self.net.append(nn.Linear(temp_input_dim, hidden_dim))
            self.net.append(nn.BatchNorm1d(hidden_dim))
            # self.net.append(nn.Dropout(p=0.1))
            self.net.append(nn.ReLU())
            temp_input_dim = hidden_dim

        # final output layer
        self.net.append(nn.Linear(hidden_dim, num_kitchens))
        self.net.append(nn.Softmax(dim=1))

        self.net = nn.Sequential(*self.net)


class CustomBatchNorm(nn.Module):
    def __init__(self, num_inputs: int, hidden_features: int):
        super(CustomBatchNorm, self).__init__()
        self.hidden_features = hidden_features
        self.linear = nn.Linear(num_inputs, hidden_features * 2)
        self.norm = nn.BatchNorm1d(hidden_features, affine=False)

    def forward(self, x: torch.Tensor, x_input: torch.Tensor) -> torch.Tensor:
        alpha_beta = self.linear(x_input)
        out = self.norm(x)
        # out = torch.repeat_interleave(alpha_beta[:, 0].unsqueeze(1), self.hidden_features, dim=1) * out
        # out = torch.repeat_interleave(alpha_beta[:, 1].unsqueeze(1), self.hidden_features, dim=1) + out
        out = alpha_beta[:, :self.hidden_features] * out + alpha_beta[:, self.hidden_features:]
        return out


class ExperimentalModel(nn.Module):
    def __init__(self, num_kitchens: int, num_items: int, hidden_layers: int = 3, hidden_features: int = 64,
                 data_means: float = 0.0):
        super(ExperimentalModel, self).__init__()
        self.data_means = data_means
        input_dim = (num_kitchens + 1) * num_items
        self.extra_norm = nn.BatchNorm1d(input_dim)
        self.first_layer = nn.Linear(input_dim, hidden_features)
        self.norm_1 = CustomBatchNorm(input_dim, hidden_features)
        self.relu_1 = nn.ReLU()

        self.hidden_layer_1 = nn.Linear(hidden_features, hidden_features)
        self.norm_2 = CustomBatchNorm(input_dim, hidden_features)
        self.relu_2 = nn.ReLU()

        # self.hidden_layer_2 = nn.Linear(hidden_features, hidden_features)
        # self.norm_3 = CustomBatchNorm(input_dim, hidden_features)
        # self.relu_3 = nn.ReLU()

        self.final_layer = nn.Linear(hidden_features, num_kitchens)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, center_data: bool = False) -> torch.Tensor:
        current_batch_size = x.shape[0]
        if center_data:
            x = x - self.data_means
        x = x.reshape(current_batch_size, -1)
        x = self.extra_norm(x)
        out = self.first_layer(x)
        out = self.norm_1(out, x)
        out = self.relu_1(out)
        out = self.relu_2(self.norm_2(self.hidden_layer_1(out), x))
        # out = self.relu_3(self.norm_3(self.hidden_layer_2(out), x))
        out = self.softmax(self.final_layer(out))

        return out
