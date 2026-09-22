"""Linear (fully connected) layer: z = xW + b."""
import numpy as np
from nn.module import Module


class Linear(Module):
    """A fully connected layer computing z = xW + b.

    Attributes:
        W (np.ndarray): weight matrix, shape (in_features, out_features).
        b (np.ndarray): bias vector, shape (out_features,).
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize the layer's weights and bias.

        Args:
            in_features (int): number of input features.
            out_features (int): number of output neurons.

        Sets:
            self.W (np.ndarray): weight matrix, shape
                (in_features, out_features). Xavier-initialized,
                not zeros (see "Weight initialization" below).
            self.b (np.ndarray): bias vector, shape
                (out_features,). Initialized to zero.
        """
        # Xavier/Glorot uniform initialization. Zeros would make every
        # unit in the layer symmetric (identical gradients during
        # backprop), so weights must start random. The scale here keeps
        # activation variance roughly stable across layers.
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, size=(in_features, out_features))
        self.b = np.zeros(out_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute this layer's output for a batch of inputs.

        Args:
            x (np.ndarray): input, shape (batch_size, in_features).

        Returns:
            np.ndarray: output, shape (batch_size, out_features).
        """
        self.x = x  # cache input; needed by backward()
        return x @ self.W + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients given the upstream gradient.

        Args:
            grad_output (np.ndarray): gradient of the loss with
                respect to this layer's output, shape (batch_size, out_features).

        Returns:
            np.ndarray: gradient of the loss with respect to this
                layer's input, shape (batch_size, in_features).

        Sets:
            self.dW (np.ndarray): gradient of the loss with
                respect to self.W, same shape as self.W.
            self.db (np.ndarray): gradient of the loss with respect
                to self.b, same shape as self.b.
        """
        # z = xW + b
        # dL/dW = x^T @ dL/dz        (in_features, out_features)
        # dL/db = sum over batch of dL/dz   (out_features,)
        # dL/dx = dL/dz @ W^T        (batch_size, in_features)
        self.dW = self.x.T @ grad_output
        self.db = grad_output.sum(axis=0)
        return grad_output @ self.W.T

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return this layer's learnable parameters.

        Returns:
            list[tuple[np.ndarray, np.ndarray]]: pairs of
            (parameter, gradient) -- [(self.W, self.dW), (self.b, self.db)].
        """
        return [(self.W, self.dW), (self.b, self.db)]
