import pytest
import torch
from modules.submodules import Sigmoid


def test_relu():
    act_fn = Sigmoid()
    x = torch.Tensor([
        [0, 0],
        [0, 0]
    ])
    expected_y = torch.Tensor([
        [0.5, 0.5],
        [0.5, 0.5]
    ])
    dJdy = torch.Tensor([
        [1, 1],
        [1, 1]
    ])
    expected_dJdx = torch.Tensor([
        [0.5*(1-0.5), 0.5*(1-0.5)],
        [0.5*(1-0.5), 0.5*(1-0.5)]
    ]) * dJdy

    y = act_fn.forward(x)
    dJdx = act_fn.backward(dJdy)

    assert torch.equal(expected_y, y)
    assert torch.equal(expected_dJdx, dJdx)
