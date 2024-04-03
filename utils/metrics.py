import torch


def compute_accuracy(true_assignments: torch.Tensor, predictions: torch.Tensor) -> float:
    accuracy = torch.sum(torch.where(true_assignments - predictions == 0, 1, 0)) / len(true_assignments) * 100
    return accuracy
