import pytest
import torch
from modules.submodules import Linear
from modules.submodules.utils import eye_init, zeros_init


IN_FEATURES = 2
OUT_FEATURES = 2


@pytest.fixture
def dataset():
    X = torch.Tensor([
        [1, 0],
        [0, 1]
    ])
    Y = torch.Tensor([
        [0, 1],
        [1, 0],
    ])

    return X, Y


@pytest.fixture
def model_eye():
    model = Linear(in_features=IN_FEATURES, out_features=OUT_FEATURES)

    model.parameters["W"] = eye_init(model.parameters["W"])
    model.parameters["b"] = zeros_init(model.parameters["b"])

    return model


@pytest.fixture
def model_rotation():
    model = Linear(in_features=IN_FEATURES, out_features=OUT_FEATURES)

    model.parameters["W"] = torch.Tensor([[0, -1], [1, 0]])
    model.parameters["b"] = zeros_init(model.parameters["b"])

    return model


@pytest.fixture
def model_bias():
    model = Linear(in_features=IN_FEATURES, out_features=OUT_FEATURES)

    model.parameters["W"] = zeros_init(model.parameters["W"])
    model.parameters["b"] = torch.ones_like(model.parameters["b"])

    return model


def test_forward_eye(dataset, model_eye):
    X, Y = dataset
    model = model_eye

    y_hat = model.forward(X)

    assert torch.allclose(X, y_hat)


def test_backward_rotation(dataset, model_rotation):
    X, _ = dataset
    model = model_rotation
    expected_y_hat = torch.Tensor([
        [0, 1],
        [-1, 0]
    ])
    dJdy = torch.Tensor([
        [1, 0],
        [0, 1]
    ])
    expected_dJdw = torch.Tensor([
        [[1, 0],
        [0, 0]],
        [[0, 0],
        [0, 1]]
    ])
    expected_dJdx = torch.Tensor([
        [0, -1],
        [1, 0]
    ])

    y_hat = model.forward(X)
    dJdx = model.backward(dJdy)
    dJdw = model.grads["dW"]

    assert torch.allclose(expected_y_hat, y_hat)
    assert torch.allclose(expected_dJdw, dJdw)
    assert torch.allclose(expected_dJdx, dJdx)


def test_forward_bias(dataset, model_bias):
    X, Y = dataset
    batch_dim = X.shape[0]
    model = model_bias
    expected_y_hat = model.parameters["b"].unsqueeze(0).repeat(batch_dim, 1)
    dJdy = torch.Tensor([
        [1, 0],
        [0, 1]
    ])
    expected_db = dJdy

    y_hat = model.forward(X)
    model.backward(dJdy)

    db = model.grads["db"]

    assert torch.allclose(expected_db, db)
    assert torch.allclose(expected_y_hat, y_hat)
