import numpy as np
import torch
from torch.utils.data import Dataset

import json


class KitchenDataset(Dataset):
    def __init__(self, data_path, num_examples: int = None):
        self.data_path = data_path

        # read json data
        f = open(data_path)
        data = json.load(f)

        N = len(data['food_data'])
        if num_examples is None or num_examples > N:
            num_examples = N

        # load input data
        # one entry consists of current setting for kitchens and food item to distribute
        # dimensions: [N, (num_kitchens + 1), num_items]
        self.food_data = torch.from_numpy(np.array(data['food_data'])[:num_examples]).type(torch.float32)
        self.food_means = torch.mean(self.food_data, dim=0)
        self.food_data = (self.food_data - self.food_means)

        # load ground truth
        # consists of one hot vectors with entries for each kitchen
        # dimensions: [N, num_kitchens]
        self.assignments = torch.from_numpy(np.array(data['kitchens_data'])[:num_examples]).type(torch.float32)

        self.num_kitchens = self.assignments.shape[-1]
        self.num_items = self.food_data.shape[-1]
        self.n = self.food_data.shape[0]

        # sanity checks
        assert self.num_kitchens + 1 == self.food_data.shape[1]
        assert self.food_data.shape[0] == self.assignments.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.food_data[idx], self.assignments[idx]

    def get_food_means(self):
        return self.food_means
