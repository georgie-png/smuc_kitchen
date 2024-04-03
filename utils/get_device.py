import torch


def get_device() -> torch.device:
    """
    Get device for training models. Returns 'cuda' if available. Otherwise returns 'cpu'
    :return: cuda if available else cpu
    """
    return torch.device("cuda" if torch.cuda.is_available() else 'cpu')
