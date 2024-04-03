import matplotlib.figure
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class Summary:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 datetime: str, num_training_examples: int, epochs: int,
                 lr: float, batch_size: int, num_kitchens: int, num_food_items: int,
                 final_accuracy: float, model_name: str = '', summary_plot: plt.Figure = None):
        """
        Custom object to save training summaries as well as trained models and optimizer state dicts
        :param model: trained model. Is saved with architecture
        :param optimizer: Adam optimizer used for training. The state dict is saved
        :param datetime: Date and time of training
        :param num_training_examples: Number of examples used for training
        :param epochs: Number of epochs
        :param lr: Learning rate
        :param batch_size: Batch size
        :param num_kitchens: Number of Kitchens
        :param num_food_items: Number of features that are tracked in every kitchen
        :param final_accuracy: Final test accuracy in percent
        :param model_name: Optional name given to the model
        :param summary_plot: Optional plot of training summary
        """
        self.model = model.to('cpu')
        self.optimizer = optimizer.state_dict()
        self.datetime = datetime
        self.num_training_examples = num_training_examples
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.num_kitchens = num_kitchens
        self.num_food_items = num_food_items
        self.final_accuracy = final_accuracy
        self.name = model_name
        self.summary_plot = summary_plot

    def __str__(self):
        as_dict = self.get_dict()
        final_string = '\n' + '=' * 32 + '\nSummary\n' + '=' * 32
        for key, value in as_dict.items():
            if not isinstance(value, dict) and not isinstance(value, np.ndarray) and not isinstance(value, nn.Module)\
                    and not isinstance(value, matplotlib.figure.Figure):
                key = key + ': ' + '.' * (28 - len(key) - 2)
                final_string += f'\n{key : <28} {value}'
        return '\n' + final_string

    def get_dict(self):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer,
            'datetime': self.datetime,
            'num_training_examples': self.num_training_examples,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'num_kitchens': self.num_kitchens,
            'num_food_items': self.num_food_items,
            'final_accuracy': self.final_accuracy,
            'model_name': self.name,
            'summary_plot': self.summary_plot
        }
