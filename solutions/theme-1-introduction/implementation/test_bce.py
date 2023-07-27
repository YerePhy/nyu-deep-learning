import pytest
import torch
from modules.submodules import bce_loss


def test_bce():
    y = torch.Tensor([
        [1, 0],
        [0, 1]
    ])
    y_hat = torch.Tensor([
        [0.8, 0.2],
        [0.8, 0.2]
    ])
    K = y.shape[-1]

    J_expected = - (y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat)).mean()
    dJ_expected = (y_hat - y) / (y_hat * (1 - y_hat) * K)

    J, dJ = bce_loss(y_hat, y)

    assert torch.allclose(dJ_expected, dJ)
    assert torch.isclose(J_expected, J)
