import torch
from typing import Tuple


__all__ = ["bce_loss"]


def bce_loss(y_hat: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor]:
    """
    Computest BCE loss and gradient.

    Args:
        y_hat: prediction.
        y: ground truth.

    Returns:
        Loss function and gradient w.r.t. ``y_hat``.
    """
    element_wise_bce = - (y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat))
    avg_bce = element_wise_bce.mean()

    K = y.shape[-1]
    dJdy_hat = (y_hat - y) / (y_hat * (1 - y_hat) * K)

    return avg_bce, dJdy_hat
