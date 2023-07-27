import pytest
import torch
from modules.submodules import mse_loss


def test_mse():
    y = torch.Tensor([
        [1, 0],
        [0, 1]
    ])
    y_hat = torch.Tensor([
        [0, 1],
        [1, 0]
    ])
    K = y.shape[-1]

    J_expected = torch.Tensor([1])
    dJ_expected = 2 * torch.Tensor([
        [-1, 1],
        [1, -1]
    ]) / K

    J, dJ = mse_loss(y_hat, y)

    assert torch.allclose(dJ_expected, dJ)
    assert torch.allclose(J_expected, J)


def test_mse_min():
    y = torch.Tensor([
        [1, 0],
        [0, 1]
    ])
    y_hat = torch.Tensor([
        [1, 0],
        [0, 1]
    ])
    K = y.shape[-1]

    J_expected = torch.Tensor([0])
    dJ_expected = torch.Tensor([
        [0, 0],
        [0, 0]
    ]) / K

    J, dJ = mse_loss(y_hat, y)

    assert torch.allclose(dJ_expected, dJ)
    assert torch.isclose(J_expected, J)
