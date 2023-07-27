import torch
from typing import Tuple, Callable
from functools import reduce, partial


__all__ = ["Sigmoid"]


class Sigmoid:
    """
    Sigmoid activation function.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor.

        Returns:
            ``x`` with sigmoid function applied element wise.
        """
        self.x = x
        self.y = 1 / (1 + torch.exp(-x))

        return self.y

    def backward(self, dJdy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dJdy: gradiento of the loss function respect to ``y``.

        Returns:
            Gradient loss function respect to ``x``
        """
        return self.y * (1 - self.y) * dJdy

