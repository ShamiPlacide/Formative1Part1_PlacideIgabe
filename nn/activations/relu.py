"""ReLU activation: zeroes out negative values."""
import numpy as np
from nn.module import Module


class ReLU(Module):
    """ReLU activation, applied elementwise: max(0, x)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute ReLU elementwise, and remember the mask.

        Args:
            x (np.ndarray): input, any shape.

        Returns:
            np.ndarray: max(0, x), elementwise, same shape as x.
        """
        # The derivative of ReLU is 1 where the input was positive and
        # 0 where it was zero or negative. This mask is cached here so
        # that backward() can apply it to the upstream gradient without
        # needing to store the full input array.
        self.mask = x > 0
        return np.where(self.mask, x, 0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients given the upstream gradient.

        Args:
            grad_output (np.ndarray): gradient of the loss with
                respect to this layer's output, same shape as the
                original input to forward.

        Returns:
            np.ndarray: gradient of the loss with respect to
            this layer's input, same shape as grad_output.
        """
        # Gradient passes through unchanged wherever the input was
        # positive, and is blocked (set to zero) wherever it was not,
        # since those positions contributed 0 to the forward output
        # regardless of how the input changes locally.
        return grad_output * self.mask
