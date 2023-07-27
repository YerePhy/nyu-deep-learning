import torch
from typing import Tuple


__all__ = ["mse_loss"]


def mse_loss(y_hat: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor]:
    """
    Computes MSE loss.

    Args:
        y_hat: prediction.
        y: ground truth.

    Returns:
        Loss function and gradient w.r.t. ``y_hat``.
    """
    K = y_hat.shape[-1]
    return torch.mean((y_hat-y)**2), 2 * (y_hat-y) / K
