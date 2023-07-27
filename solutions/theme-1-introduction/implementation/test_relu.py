import pytest
import torch
from modules.submodules import ReLU


def test_relu():
    act_fn = ReLU()
    x = torch.Tensor([
        [1, -1],
        [0, 1]
    ])
    expected_y = torch.Tensor([
        [1, 0],
        [0, 1]
    ])
    dJdy = torch.Tensor([
        [-1, 1],
        [0, 1]
    ])
    expected_dJdx = torch.Tensor([
        [-1, 0],
        [0, 1]
    ])

    y = act_fn.forward(x)
    dJdx = act_fn.backward(dJdy)

    assert torch.equal(expected_y, y)
    assert torch.equal(expected_dJdx, dJdx)
