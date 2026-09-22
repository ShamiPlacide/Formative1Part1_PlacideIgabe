"""Sigmoid activation: squashes real values into (0, 1)."""
import numpy as np
from nn.module import Module


class Sigmoid(Module):
    """Sigmoid activation, applied elementwise: 1 / (1 + e^{-x})."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute sigmoid elementwise, and remember the output.

        Args:
            x (np.ndarray): input, any shape.

        Returns:
            np.ndarray: sigmoid(x), elementwise, same shape as x.
        """
        # A direct evaluation of 1 / (1 + exp(-x)) overflows exp() for
        # strongly negative x, since -x then becomes a large positive
        # number. The standard fix is to branch on the sign of x, and
        # exponentiate only a non-positive quantity in each branch:
        #   x >= 0:  sigmoid(x) = 1 / (1 + exp(-x))       (-x <= 0)
        #   x <  0:  sigmoid(x) = exp(x) / (1 + exp(x))   ( x <  0)
        # np.where would still evaluate exp() on the full array before
        # selecting between branches, so the exponential is instead
        # computed only on the relevant subset in each case.
        positive = x >= 0
        negative = ~positive

        self.out = np.empty_like(x, dtype=float)
        self.out[positive] = 1 / (1 + np.exp(-x[positive]))

        exp_neg = np.exp(x[negative])
        self.out[negative] = exp_neg / (1 + exp_neg)

        # The output itself, rather than the input, is what backward()
        # needs, since the derivative of sigmoid can be expressed
        # entirely in terms of a = sigmoid(x): a * (1 - a). Caching a
        # here avoids recomputing the exponential in backward().
        return self.out

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
        # Local derivative of sigmoid at each point is a * (1 - a),
        # where a is the cached forward output at that point. The
        # chain rule then multiplies this elementwise by grad_output.
        local_grad = self.out * (1 - self.out)
        return grad_output * local_grad
