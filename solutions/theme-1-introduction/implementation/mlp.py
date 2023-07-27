import torch
from collections import OrderedDict


class ReLU:
    """
    Computes ReLU activation function.
    """
    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: input tensor.

        Returns:
            ReLU applied over the input tensor.
        """
        return torch.where(x > 0, x, 0)

    def backward(self, dJdy, x) -> torch.Tensor:
        """
        Args:
            dJdy: gradient w.r.t. forward output.
            x: cached forward input.

        Returns:
            Gradient of loss function w.r.t. input tensor `x`.
        """
        return torch.where(x > 0, dJdy, 0)


class Sigmoid:
    """
    Sigmoid activation function.
    """

    def __init__(self):
        pass

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: input tensor.

        Returns:
            ``x`` with sigmoid function applied element wise.
        """
        return 1 / (1 + torch.exp(-x))

    def backward(self, dJdy, x) -> torch.Tensor:
        """
        Args:
            dJdy: gradient w.r.t. forward output.
            x: cached forward input.

        Returns:
            Gradient of loss function w.r.t. input tensor `x`.
        """
        sigmoid = 1 / (1 + torch.exp(-x))
        return dJdy * sigmoid * (1 - sigmoid)


class Identity:
    """
    Identity activation function.
    """

    def __init__(self):
        pass

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: input tensor.

        Returns:
            ``x`` with sigmoid function applied element wise.
        """
        return x

    def backward(self, dJdy, x):
        """
        Args:
            dJdy: gradient w.r.t. forward output.
            x: cached forward input.

        Returns:
            Gradient of loss function w.r.t. input tensor `x`.
        """
        return dJdy


class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = self._get_activation_fn(f_function)
        self.g_function = self._get_activation_fn(g_function)

        self.parameters = OrderedDict([
            ("W1", torch.randn(linear_1_out_features, linear_1_in_features)),
            ("b1", torch.randn(linear_1_out_features)),
            ("W2", torch.randn(linear_2_out_features, linear_2_in_features)),
            ("b2", torch.randn(linear_2_out_features)),
        ])
        self.grads = OrderedDict([
            ("dJdW1", torch.zeros(linear_1_out_features, linear_1_in_features)),
            ("dJdb1", torch.zeros(linear_1_out_features)),
            ("dJdW2", torch.zeros(linear_2_out_features, linear_2_in_features)),
            ("dJdb2", torch.zeros(linear_2_out_features)),
        ])

        # put all the cache value you need in self.cache
        self.cache = dict()

    def _get_activation_fn(self, act_fn_name):
        """
        Args:
            act_fn_name: string for the f function: relu | sigmoid | identity

        Returns:
            An instance of the activation function specified.
        """
        if act_fn_name == "relu":
            return ReLU()
        elif act_fn_name == "sigmoid":
            return Sigmoid()
        elif act_fn_name == "identity":
            return Identity()
        else:
            raise NotImplementedError

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)

        Returns:
            The inference of the model.
        """
        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]

        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]

        z1 = torch.einsum('ij,kj->ki', W1, x) + b1
        z2 = self.f_function.forward(z1)
        z3 = torch.einsum('ij,kj->ki', W2, z2) + b2
        y_hat = self.g_function.forward(z3)

        self.cache["x"] = x
        self.cache["z1"] = z1
        self.cache["z2"] = z2
        self.cache["z3"] = z3
        self.cache["y_hat"] = y_hat

        return y_hat

    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]

        x = self.cache["x"]
        z1 = self.cache["z1"]
        z2 = self.cache["z2"]
        z3 = self.cache["z3"]

        dJdz3 = self.g_function.backward(dJdy_hat, z3)
        dJdz2 = torch.einsum('ij,kj->ki', W2.T, dJdz3)
        dJdz1 = self.f_function.backward(dJdz2, z1)

        self.grads["dJdW2"] = torch.einsum('ij,ik->ikj', z2, dJdz3).mean(0)
        self.grads["dJdb2"] = dJdz3.mean(0)
        self.grads["dJdW1"] = torch.einsum('ij,ik->ikj', x, dJdz1).mean(0)
        self.grads["dJdb1"] = dJdz1.mean(0)

    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()


def mse_loss(y, y_hat):
    """
    Computes MSE loss.

    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    K = y_hat.shape[-1]
    return torch.mean((y_hat-y)**2), 2 * (y_hat-y) / K


def bce_loss(y, y_hat):
    """
    Computes BCE loss.

    Args:
        y_hat: the prediction tensor
        y: the label tensor

    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    element_wise_bce = - (y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat))
    avg_bce = element_wise_bce.mean()

    K = y.shape[-1]
    dJdy_hat = (y_hat - y) / (y_hat * (1 - y_hat) * K)

    return avg_bce, dJdy_hat

