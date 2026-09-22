"""Train a single-layer network (Linear + Sigmoid + BCE) on a toy dataset."""

import numpy as np

from nn.activations import Sigmoid
from nn.layers import Linear
from nn.losses import CrossEntropyLoss
from nn.optim import SGD


def toy_data() -> tuple[np.ndarray, np.ndarray]:
    """Return the AND-gate toy dataset.

    Returns:
        tuple[np.ndarray, np.ndarray]: (X, y). X has shape (4, 2),
            y has shape (4, 1), values 0 or 1.
    """
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [0], [0], [1]], dtype=float)
    return x, y


def train(epochs: int = 4000, lr: float = 1.0, seed: int = 0) -> list[float]:
    """Train a Linear + Sigmoid model with SGD on the toy dataset.

    Args:
        epochs (int): number of training iterations.
        lr (float): learning rate for SGD.
        seed (int): random seed for reproducible weight init.

    Returns:
        list[float]: the loss recorded at every epoch, in order.
    """
    np.random.seed(seed)
    x, y = toy_data()

    linear = Linear(in_features=2, out_features=1)
    sigmoid = Sigmoid()
    loss_fn = CrossEntropyLoss()
    optimizer = SGD([], lr=lr)

    history = []
    for _ in range(epochs):
        z = linear.forward(x)
        a = sigmoid.forward(z)
        loss = loss_fn.forward(a, y)
        history.append(loss)

        grad = loss_fn.backward()
        grad = sigmoid.backward(grad)
        linear.backward(grad)

        optimizer.parameters = linear.parameters()
        optimizer.step()
        optimizer.zero_grad()

    global _model
    _model = (linear, sigmoid)
    return history


def accuracy(loss_history: list[float] = None) -> float:
    """Report final classification accuracy on the toy dataset.

    Args:
        loss_history (list[float]): unused; kept for interface
            compatibility. Retrains with default settings if no
            model has been trained yet.

    Returns:
        float: fraction of correctly classified examples, in [0, 1].
    """
    if "_model" not in globals():
        train()
    linear, sigmoid = _model
    x, y = toy_data()
    predictions = sigmoid.forward(linear.forward(x))
    predicted_labels = (predictions >= 0.5).astype(float)
    return float(np.mean(predicted_labels == y))


if __name__ == "__main__":
    loss_history = train()
    for epoch in (0, 999, 1999, 2999, 3999):
        print(f"epoch {epoch}: loss = {loss_history[epoch]:.4f}")
    print(f"final accuracy: {accuracy():.2%}")
