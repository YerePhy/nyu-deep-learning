import torch
from typing import Callable, Any


__all__ = [
    "eye_init",
    "zeros_init"
]


def eye_init(x: torch.Tensor) -> torch.Tensor:
    """
    Initializes weights to identity.

    Args:
        x: weights.

    Returns:
        Identity with the same shape
        and type as the input tensor.
    """
    return torch.eye(*x.size(), out=torch.empty_like(x))


def zeros_init(x: torch.Tensor) -> torch.Tensor:
    """
    Initializes weights to zeros.

    Args:
        x: weights.

    Returns:
        Zeros with the same shape
        and type as the input tensor.
    """
    return torch.zeros_like(x)
