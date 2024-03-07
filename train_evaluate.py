import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm

from models import SimpleMLP
from data_loaders import KitchenDataset


def get_device() -> torch.device:
    """
    Get device for training models. Returns 'cuda' if available. Otherwise returns 'cpu'
    :return: cuda if available else cpu
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


def initialize_model(num_kitchens: int, num_items: int, data_means: torch.Tensor,
                     hidden_layers: int = 1, hidden_features: int = 64,
                     device: torch.device = 'cpu') -> SimpleMLP:
    """
    Initialize simple MLP model
    :param num_kitchens: Number of kitchens
    :param num_items: Number of food items / sensor data. Number of features per kitchen
    :param hidden_layers: Number of hidden layers
    :param hidden_features: Number of hidden features per layer
    :param data_means: Mean values of food data. Dimensions [num_kitchens + 1, num_items]
    :param device: device for training
    :return: SimpleMLP model
    """
    model = SimpleMLP(num_kitchens=num_kitchens,
                      num_items=num_items,
                      hidden_layers=hidden_layers,
                      hidden_features=hidden_features,
                      data_means=data_means).to(device)
    return model


def train_model(data_path: str, lr: float = 0.0008, batch_size: int = 512, num_epochs: int = 300,
                hidden_layers: int = 1, hidden_features: int = 64, final_plot: bool = False) -> tuple[SimpleMLP, dict]:
    """
    Train SimpleMLP. Model is initialized with given parameters. The remaining dimensions are inferred from the data.
    The data is split into training and test data with an 80%-20% split.
    :param data_path: Path to .json data
    :param lr: learning rate
    :param batch_size: batch size
    :param num_epochs: number of training epochs
    :param hidden_layers: number of hidden layers of model
    :param hidden_features: number of features per hidden layer
    :param final_plot: If True, plot a summary of the training parameters
    :return: trained model and dict with training summary containing the state dict of the optimizer
    """
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    device = get_device()

    # get full dataset
    kitchen_dataset = KitchenDataset(data_path)
    food_means = kitchen_dataset.get_food_means()

    # split data into training and test data
    num_kitchens = kitchen_dataset.num_kitchens
    num_items = kitchen_dataset.num_items
    num_training_examples = int(0.8 * len(kitchen_dataset))
    num_test_examples = len(kitchen_dataset) - num_training_examples

    train_data, test_data = torch.utils.data.random_split(kitchen_dataset, [num_training_examples, num_test_examples])

    # setup data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # initialize optimizer
    model = initialize_model(num_kitchens=num_kitchens,
                             num_items=num_items,
                             data_means=food_means,
                             hidden_layers=hidden_layers,
                             hidden_features=hidden_features,
                             device=device)
    optim = optimizers.Adam(model.parameters(), lr=lr)
    criterion = F.cross_entropy

    # TRAIN CLASSIFIER
    print('=' * 32)
    print(f'Start training')
    print(f'Device: {device}')
    print(f'Number of epochs: {num_epochs}')
    print(f'Number of training examples: {num_training_examples}')
    print('=' * 32)

    losses = []
    test_losses = []
    accuracies = []
    test_x = []

    # -----------------------------------------------------------------
    # MAIN TRAINING LOOP
    # -----------------------------------------------------------------
    prog_bar = tqdm(range(num_epochs))
    num_training_iterations = 0
    for epoch in range(num_epochs):
        # train one epoch
        for _, (data, targets) in enumerate(iter(train_loader)):
            data, targets = data.to(device), targets.to(device)

            prediction = model(data)
            loss = criterion(prediction, targets)
            losses.append(loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()

            num_training_iterations += 1

        # validation
        test_loss = 0
        test_predictions = []
        true_labels = []
        with torch.no_grad():
            # iterate through test set
            for _, (data, targets) in enumerate(iter(test_loader)):
                data, targets = data.to(device), targets.to(device)
                prediction = model(data)
                test_loss += criterion(prediction, targets, reduction='sum')
                test_predictions.append(prediction)
                true_labels.append(targets)
            # average test loss
            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)
            test_x.append(num_training_iterations)

            # compute accuracy
            true_test_labels = torch.argmax(torch.cat(true_labels).detach().cpu(), dim=1)
            all_predictions = torch.argmax(torch.cat(test_predictions).detach().cpu(), dim=1)
            accuracy = torch.sum(torch.where(all_predictions - true_test_labels == 0, 1, 0)) / len(test_data) * 100
            accuracies.append(accuracy)

        prog_bar.update(1)

    prog_bar.close()

    # plot summary
    if final_plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle('Training Summary')

        ax[0].plot(np.log(losses), label='Training Loss')
        ax[0].plot(test_x, np.log(test_losses), label='Test Loss')
        ax[0].legend()
        ax[0].set_xlabel('Iterations')
        ax[0].set_ylabel('log loss')
        ax[0].set_title(f'Log loss over {num_epochs} epochs')

        ax[1].plot(test_x, accuracies, label='Test Accuracy')
        ax[1].set_xlabel('Iterations')
        ax[1].set_ylabel('Accuracy %')
        ax[1].set_title(f'Final Accuracy: {accuracies[-1] :.2f}%')

        plt.show()

    summary = {
        'datetime': current_date,
        'num_training_examples': num_training_examples,
        'epochs': num_epochs,
        'lr': lr,
        'batch_size': batch_size,
        'num_kitchens': num_kitchens,
        'num_items': num_items,
        'final_accuracy': accuracies[-1],
        'optimizer_state_dict': optim.state_dict(),
    }

    return model, summary


def save_model_summary(model: nn.Module, summary: dict, path: str) -> None:
    """
    Save model including architecture and corresponding training summary
    :param model: Model to save
    :param summary: Training summary
    :param path: where to save
    """
    summary['model'] = model.to('cpu')
    torch.save(summary, path)


def load_model_for_inference(path: str) -> tuple[SimpleMLP, dict]:
    """
    Load model from saved checkpoint. Model is loaded to cpu and put into eval mode
    :param path: path to checkpoint file
    :return: model (SimpleMLP), summary (dict)
    """
    summary = torch.load(path, map_location='cpu')
    model = summary['model']
    model.eval()
    return model, summary


def load_model_and_optimizer(path: str) -> tuple[SimpleMLP, optimizers.Adam, dict]:
    """
    Load model and optimizer from saved checkpoint.
    Can be used to continue training
    :param path: path to checkpoint file
    :return: model (SimpleMLP), optimizer (Adam) and training summary (dict)
    """
    summary = torch.load(path, map_location='cpu')
    model = summary['model']
    optim = optimizers.Adam(params=model.parameters())
    optim.load_state_dict(summary['optimizer_state_dict'])
    return model, optim, summary


def print_summary(summary: dict) -> None:
    """
    Print the summary
    :param summary: dictionary with summary information
    """
    print('\n')
    print('=' * 32)
    print('Summary')
    print('=' * 32)
    for key, value in summary.items():
        if not isinstance(value, dict) and not isinstance(value, np.ndarray) and not isinstance(value, nn.Module):
            key = key + ': ' + '.' * (28 - len(key) - 2)
            print(f'{key : <28} {value}')
    print('\n')


def classify(model: SimpleMLP, x: np.ndarray | torch.Tensor | list, res_format: str = '') -> np.ndarray:
    """

    :param model: trained Model
    :param x: Input data to classify. Dimensions are [batch_size, num_kitchens + 1, num_items] for multiple
    classifications or [num_kitchens + 1, num_items] for a single classification.
    :param res_format: How to format the results. Can be one of {'', 'one_hot', 'label'}
    default: results contain assignment probabilities between 0-1.
    'one_hot': results contain one hot vectors, with a one marking the chosen kitchen.
    'label': results contain kitchen labels. [4, 0, 9] means the first item goes to kitchen 4, the second to kitchen 0
            and the third to kitchen 9.

    :return: For a single classification, the output has dimensions [num_kitchens] (default and 'one_hot' encoding)
    or [1] when direct labelling is applied.
    If multiple data points are passed, the output has dimensions [batch_size, num_kitchens] resp. [batch_size]
    """
    # convert input to torch tensor
    if isinstance(x, list):
        x = torch.tensor(x)
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    # if input is 2D, we add first dimension
    # now x should be [N, num_kitchens + 1, num_items]
    if x.ndim == 2:
        x = x.unsqueeze(0)

    # classify
    with torch.no_grad():
        prediction = model(x.type(torch.float32), center_data=True)

    # adjust format of prediction
    if res_format == 'label':
        return torch.argmax(prediction, dim=1).squeeze().numpy()
    elif res_format == 'one_hot':
        res = F.one_hot(torch.argmax(prediction, dim=1), num_classes=x.shape[1] - 1).squeeze().numpy()
        return res
    else:
        return prediction.squeeze().numpy()
