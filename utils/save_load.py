import torch

from utils.summary import Summary


def save_summary(path: str, summary: Summary):
    torch.save(summary, path)
    print(f'Saved summary to: {path}')


def load_summary(path: str) -> Summary:
    summary = torch.load(path, map_location='cpu')
    print(f'Loaded summary from: {path}')
    return summary


def load_model_for_inference(path: str) -> tuple[torch.nn.Module, Summary]:
    """
    Load model from saved checkpoint. Model is loaded to cpu and put into eval mode
    :param path: path to checkpoint file
    :return: model, summary object
    """
    summary = load_summary(path)
    model = summary.model
    model.eval()
    return model, summary


def load_model_and_optimizer(path: str) -> tuple[torch.nn.Module, torch.optim.Optimizer, Summary]:
    """
    Load model and optimizer from saved checkpoint.
    Can be used to continue training
    :param path: path to checkpoint file
    :return: model, optimizer (Adam) and training summary
    """
    summary = load_summary(path)
    model = summary.model
    optimizer = torch.optim.Adam(params=model.parameters())
    optimizer.load_state_dict(summary.optimizer)
    return model, optimizer, summary
