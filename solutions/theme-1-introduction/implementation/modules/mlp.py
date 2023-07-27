import torch
from collections import OrderedDict
from typing import NewType
from .submodules import Linear


__all__ = ["MLP"]


ActivationFn = NewType('ActivationFn', object)


class MLP:
    """
    Multilayer perceptron with 2 blocks of Linear + Activation.
    """

    def __init__(
        self,
        linear_1_in_features: int,
        linear_1_out_features: int,
        linear_2_in_features: int,
        linear_2_out_features: int,
        f_function: ActivationFn,
        g_function: ActivationFn
    ) -> None:
        """
        Args:
            linear_1_in_features: linear layer 1 input features.
            linear_1_out_features: linear layer 2 output features.
            linear_2_in_features: linear layer 2 input features.
            linear_2_out_features: linear layer 2 output features.
            f_function: first activation function.
            g_function: last activation function.
        """
        self.linear_1_in_features = linear_1_in_features
        self.linear_1_out_features = linear_1_out_features
        self.linear_2_in_features = linear_2_in_features
        self.linear_2_out_features = linear_2_out_features
        self.f_function = f_function
        self.g_function = g_function
        self.submodules = OrderedDict([
            ("linear_1", Linear(linear_1_in_features, linear_1_out_features)),
            ("f", f_function),
            ("linear_2", Linear(linear_2_in_features, linear_2_out_features)),
            ("g", g_function)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor.
        """
        for submodule in self.submodules.values():
            x = submodule.forward(x)

        return x

    def backward(self, dJ) -> torch.Tensor:
        for submodule in reversed(self.submodules.values()):
            dJ = submodule.backward(dJ)
