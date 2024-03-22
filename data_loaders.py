import numpy as np
import torch
from torch.utils.data import Dataset
#from sklearn.decomposition import TruncatedSVD

import json


def normalize_data(data: torch.Tensor):
    """
    Normalize input data. If the data range is 0, the value is set to -1.
    :param data: Input data to normalize
    :return: Normalized data, maximum values, min values
    """
    maxima = torch.max(data, dim=0)[0]
    minima = torch.min(data, dim=0)[0]
    data_range = (maxima - minima)
    data_range[data_range == 0] = 1.0
    normalized_data = (data - minima) / data_range
    normalized_data = (normalized_data - 0.5) * 2.0
    return normalized_data, maxima, minima


class KitchenDataset(Dataset):
    def __init__(self, data_path, num_examples: int = None, train: str = ''):
        self.data_path = data_path

        # read json data
        f = open(data_path)
        data = json.load(f)

        if train == 'train':
            food_tag = 'food_data_train'
            kitchen_tag = 'kitchens_data_train'
        elif train == 'test':
            food_tag = 'food_data_test'
            kitchen_tag = 'kitchens_data_test'
        else:
            food_tag = 'food_data'
            kitchen_tag = 'kitchens_data'

        N = len(data[food_tag])
        if num_examples is None or num_examples > N:
            num_examples = N

        # load input data
        # one entry consists of current setting for kitchens and food item to distribute
        # dimensions: [N, (num_kitchens + 1), num_items]
        self.food_data = torch.from_numpy(np.array(data[food_tag])[:num_examples]).type(torch.float32)
        #self.food_means = torch.mean(self.food_data, dim=0)

        if train != 'test':
            self.food_data, self.data_max, self.data_min = normalize_data(self.food_data)

        # load ground truth
        # consists of one hot vectors with entries for each kitchen
        # dimensions: [N, num_kitchens]
        self.assignments = torch.from_numpy(np.array(data[kitchen_tag])[:num_examples]).type(torch.float32)

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

    def get_max_min(self):
        return self.data_max, self.data_min

    def get_food_means(self):
        return 0 #self.food_means


'''
class KitchenDatasetSVD(Dataset):
    def __init__(self, data_path, n_components: int = 30):
        self.data_path = data_path
        self.n_components = n_components

        # read json data
        f = open(data_path)
        data = json.load(f)

        N = len(data['food_data'])

        # load input data
        # one entry consists of current setting for kitchens and food item to distribute
        # dimensions: [N, (num_kitchens + 1), num_items]
        self.food_data = np.array(data['food_data'])
        self.food_data = self.food_data.reshape(N, -1)
        self.svd = TruncatedSVD(n_components=self.n_components)
        self.svd.fit(self.food_data)
        self.food_data = torch.from_numpy(self.svd.transform(self.food_data)).type(torch.float32)

        # load ground truth
        # consists of one hot vectors with entries for each kitchen
        # dimensions: [N, num_kitchens]
        self.assignments = torch.from_numpy(np.array(data['kitchens_data'])).type(torch.float32)

        self.num_kitchens = self.assignments.shape[-1]
        self.num_items = self.food_data.shape[-1]
        self.n = self.food_data.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.food_data[idx], self.assignments[idx]

    def get_food_means(self):
        return 0

    def get_svd(self):
        return self.svd
'''


def split_and_save_data(data_path, destination, shuffle: bool = True):
    f = open(data_path)
    data = json.load(f)

    N = len(data['food_data'])
    num_training_examples = int(0.8 * N)
    num_test_examples = N - num_training_examples

    print(f'Number of training examples: {num_training_examples}')
    print(f'Number of test examples: {num_test_examples}')

    if shuffle:
        indices = np.random.permutation(N)
    else:
        indices = np.arange(N)

    food_data = np.array(data['food_data'])[indices]
    assignments = np.array(data['kitchens_data'])[indices]

    x_train = food_data[:num_training_examples]
    y_train = assignments[:num_training_examples]

    x_test = food_data[num_training_examples:]
    y_test = assignments[num_training_examples:]

    np.save(destination + '_x_train.npy', x_train)
    np.save(destination + '_y_train.npy', y_train)
    np.save(destination + '_x_test.npy', x_test)
    np.save(destination + '_y_test.npy', y_test)


class KitchenDatasetNumpy(Dataset):
    def __init__(self, data_path, train: bool):
        if train:
            self.food_data = np.load(data_path + '_x_train.npy')
            self.assignments = np.load(data_path + '_y_train.npy')

            print(f'loaded {len(self.food_data)} training examples from {data_path + "_x_train.npy"}')
            print(f'loaded {len(self.assignments)} training assignments from {data_path + "_y_train.npy"}')
        else:
            self.food_data = np.load(data_path + '_x_test.npy')
            self.assignments = np.load(data_path + '_y_test.npy')

            print(f'loaded {len(self.food_data)} test examples from {data_path + "_x_test.npy"}')
            print(f'loaded {len(self.assignments)} test assignments from {data_path + "_y_test.npy"}')

        self.food_data = torch.from_numpy(self.food_data).type(torch.float32)

        if train is True:
            self.food_data, self.maxima, self.minima = normalize_data(self.food_data)
        else:
            self.maxima = 1
            self.minima = 0

        self.assignments = torch.from_numpy(self.assignments).type(torch.float32)

        self.num_kitchens = self.assignments.shape[-1]
        self.num_items = self.food_data.shape[-1]

    def __len__(self):
        return len(self.food_data)

    def __getitem__(self, idx):
        return self.food_data[idx], self.assignments[idx]

    def get_max_min(self):
        return self.maxima, self.minima
