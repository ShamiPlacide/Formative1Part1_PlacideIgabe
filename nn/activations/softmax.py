"""Softmax activation: converts logits into a probability distribution."""
import numpy as np
from nn.module import Module


class Softmax(Module):
    """Softmax activation, applied row-wise to a batch of logits.

    Unlike ReLU or Sigmoid, each output depends on every logit in its
    own row, not just the matching input; see "A shape subtlety" above.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities for a batch of logits.

        Args:
            x (np.ndarray): logits, shape (batch_size, C).

        Returns:
            np.ndarray: probabilities, shape (batch_size, C).
            Each row sums to 1.
        """
        # Max-subtraction trick: softmax(x) equals softmax(x - max(x)),
        # since subtracting a constant from every logit in a row cancels
        # out in the ratio of exponentials, but keeps exp() from
        # overflowing on large logits.
        row_max = np.max(x, axis=1, keepdims=True)
        exp_shifted = np.exp(x - row_max)
        row_sum = np.sum(exp_shifted, axis=1, keepdims=True)
        self.out = exp_shifted / row_sum  # cache output; needed by backward()
        return self.out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients given the upstream gradient.

        Args:
            grad_output (np.ndarray): gradient of the loss with
                respect to this layer's output, shape (batch_size, C).

        Returns:
            np.ndarray: gradient of the loss with respect to
            this layer's input (the logits), shape (batch_size, C).
        """
        # For a single row, the softmax Jacobian J has entries
        #   J[i, j] = out[i] * (delta_ij - out[j]),
        # where delta_ij is 1 when i equals j and 0 otherwise. The
        # gradient with respect to the input is grad_output @ J, which
        # expands, per row, to:
        #   dx[i] = out[i] * (grad_output[i] - sum_j grad_output[j] * out[j])
        #
        # Rather than constructing the full (C, C) Jacobian for every
        # row, this identity is evaluated directly, which is both
        # simpler and considerably cheaper.
        weighted_sum = np.sum(grad_output * self.out, axis=1, keepdims=True)
        grad_input = self.out * (grad_output - weighted_sum)
        return grad_input
