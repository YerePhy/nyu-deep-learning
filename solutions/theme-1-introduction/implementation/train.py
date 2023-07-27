import torch
import matplotlib.pyplot as plt
from modules import MLP
from modules.submodules import Sigmoid, ReLU, Linear, mse_loss, bce_loss

# Hyperparameters
EPOCHS = 300
LR = 0.1
LOSS = "BCE"  # BCE or MSE

# Dummy dataset
X = torch.Tensor([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],

])
Y = torch.Tensor([
    [1],
    [1],
    [1],
    [1],
    [1],
    [0],
    [0],
    [0],
    [0],
    [0]
])

# model and loss
model = MLP(
    linear_1_in_features=10,
    linear_1_out_features=20,
    linear_2_in_features=20,
    linear_2_out_features=1,
    f_function=ReLU(),
    g_function=Sigmoid()
)
loss = {
    "BCE": bce_loss,
    "MSE": mse_loss
}[LOSS]

# training loop
epochs = [n for n in range(EPOCHS)]
loss_values = []

for n in epochs:
    y_hat = model.forward(X)
    j, dj = loss(y_hat, Y)

    dj = model.backward(dj)

    # update weights with mini batch gradient descent
    for submodule in model.submodules.values():
        if isinstance(submodule, Linear):
            submodule.parameters["W"] = submodule.parameters["W"] - LR * submodule.grads["dW"].mean(0)
            submodule.parameters["b"] = submodule.parameters["b"] - LR * submodule.grads["db"].mean(0)

    loss_values.append(j)

print(f"FINAL PRED:\n{model.forward(X)}\n GROUND_TRUTH:\n{Y})")

plt.plot(epochs, loss_values)
plt.title(f"Loss: {LOSS}, LR: {LR}")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
