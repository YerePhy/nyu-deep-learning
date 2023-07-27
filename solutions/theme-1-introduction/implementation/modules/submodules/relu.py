import torch
from typing import Tuple


__all__ = ["ReLU"]


class ReLU:
    """
    Computes ReLU activation function.

    .. code-block::

        y = torch.where(x > 0, x, 0)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor.

        Returns:
            ReLU applied over the input tensor.
        """
        self.x = x
        return torch.where(self.x > 0, self.x, 0)

    def backward(self, dJdy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dJdy: gradient w.r.t. the output of
                the forward pass.

        Returns:
            Gradient of loss function w.r.t. input tensor `x`.
        """
        return torch.where(self.x > 0, dJdy, 0)

