import torch
from typing import Tuple
from collections import OrderedDict


__all__ = ["Linear"]


class Linear:
    """
    Linear layer ``Y = WX + b``.
    """
    def __init__(self, in_features: int, out_features: int):
        """
        Args:
            in_features: input dims.
            out_features: output dims.

        """
        self.in_features = in_features
        self.out_features = out_features

        self.parameters = OrderedDict([
            ("W", torch.randn(out_features, in_features)),
            ("b", torch.randn(out_features))
        ])

        self.grads = OrderedDict([
            ("dW", torch.zeros(out_features, in_features)),
            ("db", torch.zeros(out_features))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor.

        Returns:
            The result of ``Y = WX + b``.

        """
        W = self.parameters["W"]
        b = self.parameters["b"]

        self.x = x
        self.y = torch.einsum('ij,kj->ki', W, x) + b

        return self.y

    def backward(self, dJdy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dJdy: gradient of the loss respect to ``Y``.

        Returns:
            The gradiento of J respect to the input tensor ``x```.
        """
        W = self.parameters["W"]

        self.grads["dW"] = torch.einsum('ij,ik->ikj', self.x, dJdy)
        self.grads["db"] = dJdy

        dJdx = torch.einsum('ij,kj->ki', W.T, dJdy)

        return dJdx

