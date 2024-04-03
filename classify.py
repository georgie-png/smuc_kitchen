import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def classify(model: nn.Module, x: np.ndarray | torch.Tensor | list, res_format: str = '') -> np.ndarray:
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