import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm

import utils.save_load
from models import SimpleMLP, MLPVariableLayers, SimplestMLP
from data_loaders import KitchenDataset

from utils.summary import Summary
from utils.metrics import compute_accuracy
from utils.get_device import get_device
from utils.save_load import load_model_and_optimizer


def get_list_of_models() -> list[str]:
    """
    Returns a list of available model architectures that can be trained with train_model
    :return: list of available model architectures
    """
    return ['default', 'MLP_batch_norm']


def train_model(data_paths: list, lr: float = 1e-4, epochs: int = 100, layers: list = [16, 6, 16],
                model_type: str = 'default') -> tuple[nn.Module, Summary]:
    """
    Train a simple MLP on given data
    :param data_paths: List of json files containing training and test data
    :param lr: Learning rate. Default is 1e-4
    :param epochs: Number of epochs. Default is 100
    :param layers: List of layers. Every entry of the list is the number of features in the corrensponding layer
    :param model_type: Model architecture. Currently only the default fully connected MLP is supported
    :return: Trained model and summary object
    """
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    device = get_device()

    # Additional hyperparameters
    batch_size = 512
    # num_training_examples = 1000

    # load dataset
    train_data = KitchenDataset(data_paths, train='train')  # , num_examples=num_training_examples)
    test_data = KitchenDataset(data_paths, train='test')

    print(f'Total number of training examples: {len(train_data)}')
    print(f'Total number of test examples: {len(test_data)}')

    # setup data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # get data range
    data_max, data_min = train_data.get_max_min()

    # initialize model
    if model_type == 'default':
        '''
        Default. Fully connected MLP without any batch norm or dropout.
        '''
        model = SimplestMLP(num_kitchens=train_data.num_kitchens,
                            num_items=train_data.num_items,
                            hidden_features=layers,
                            data_max=data_max,
                            data_min=data_min).to(device)
    elif model_type == 'MLP_batch_norm':
        '''
        Fully connected MLP with batch norm before ReLU.
        '''
        model = MLPVariableLayers(num_kitchens=train_data.num_kitchens,
                                  num_items=train_data.num_items,
                                  hidden_features=layers,
                                  data_max=data_max,
                                  data_min=data_min).to(device)
    else:
        print(f'No model type found for \"{model_type}\"')
        return None, None

    # initialize optimizer
    optim = optimizers.Adam(model.parameters(), lr=lr)
    criterion = F.cross_entropy

    # TRAIN CLASSIFIER
    print('=' * 32)
    print(f'Start training')
    print(f'Device: {device}')
    print(f'Number of epochs: {epochs}')
    print(f'Number of training examples: {len(train_data)}')
    print('=' * 32)

    losses = []
    test_losses = []
    accuracies = []
    test_x = []

    # -----------------------------------------------------------------
    # MAIN TRAINING LOOP
    # -----------------------------------------------------------------
    prog_bar = tqdm(range(epochs))
    num_training_iterations = 0
    for epoch in range(epochs):
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
                prediction = model(data, center_data=True)
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
            accuracy = compute_accuracy(true_test_labels, all_predictions)
            accuracies.append(accuracy)

        prog_bar.update(1)
    prog_bar.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Training Summary')

    ax[0].plot(np.log(losses), label='Training Loss')
    ax[0].plot(test_x, np.log(test_losses), label='Test Loss')
    ax[0].legend()
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('log loss')
    ax[0].set_title(f'Log loss over {epochs} epochs')

    ax[1].plot(test_x, accuracies, label='Test Accuracy')
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Test accuracy %')
    ax[1].set_title(f'Final Accuracy: {accuracies[-1] :.2f}%')

    summary = Summary(
        model=model,
        optimizer=optim,
        datetime=current_date,
        num_training_examples=len(train_data),
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        num_kitchens=train_data.num_kitchens,
        num_food_items=train_data.num_items,
        final_accuracy=accuracies[-1],
        model_name='',
        summary_plot=fig
    )
    plt.close(fig)

    return model.eval(), summary


def continue_training(summary: str | Summary, data_paths: list, epochs: int) -> tuple[nn.Module, Summary]:
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    device = get_device()

    # Additional hyperparameters
    batch_size = 512
    # num_training_examples = 1000

    # load dataset
    train_data = KitchenDataset(data_paths, train='train')  # , num_examples=num_training_examples)
    test_data = KitchenDataset(data_paths, train='test')

    print(f'Total number of training examples: {len(train_data)}')
    print(f'Total number of test examples: {len(test_data)}')

    # setup data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # get data range
    data_max, data_min = train_data.get_max_min()

    # Load model and optimizer
    # either load them from file or extract them from summary
    if isinstance(summary, str):
        model, optim, loaded_summary = load_model_and_optimizer(summary)
    elif isinstance(summary, Summary):
        model = summary.model
        optim = optimizers.Adam(model.parameters())
        optim.load_state_dict(summary.optimizer)
        loaded_summary = summary
    model.train()
    model.to(device)
    criterion = F.cross_entropy

    # CONTINUE TRAINING
    print('=' * 32)
    print(f'Start training')
    print(f'Device: {device}')
    print(f'Number of epochs: {epochs}')
    print(f'Number of training examples: {len(train_data)}')
    print('=' * 32)

    losses = []
    test_losses = []
    accuracies = []
    test_x = []

    # -----------------------------------------------------------------
    # MAIN TRAINING LOOP
    # -----------------------------------------------------------------
    prog_bar = tqdm(range(epochs))
    num_training_iterations = 0
    for epoch in range(epochs):
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
                prediction = model(data, center_data=True)
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
            accuracy = compute_accuracy(true_test_labels, all_predictions)
            accuracies.append(accuracy)

        prog_bar.update(1)
    prog_bar.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Training Summary')

    ax[0].plot(np.log(losses), label='Training Loss')
    ax[0].plot(test_x, np.log(test_losses), label='Test Loss')
    ax[0].legend()
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('log loss')
    ax[0].set_title(f'Log loss over {epochs} epochs')

    ax[1].plot(test_x, accuracies, label='Test Accuracy')
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Test accuracy %')
    ax[1].set_title(f'Final Accuracy: {accuracies[-1] :.2f}%')

    updated_summary = Summary(
        model=model,
        optimizer=optim,
        datetime=current_date,
        num_training_examples=len(train_data),
        epochs=epochs,
        lr=loaded_summary.lr,
        batch_size=batch_size,
        num_kitchens=train_data.num_kitchens,
        num_food_items=train_data.num_items,
        final_accuracy=accuracies[-1],
        model_name=loaded_summary.name,
        summary_plot=fig
    )
    plt.close(fig)

    return model.eval(), updated_summary
