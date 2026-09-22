"""Binary cross-entropy loss."""
import numpy as np


class CrossEntropyLoss:
    """Binary cross-entropy loss for a single output probability.

    Does not subclass Module; see the note below.
    """

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute the average binary cross-entropy loss.

        Args:
            predictions (np.ndarray): predicted probabilities,
                shape (m,) or (m, 1). Clip away from exactly
                0 or 1 before use; see "Numerical stability" above.
            targets (np.ndarray): true labels, same shape as
                predictions, values 0 or 1.

        Returns:
            float: the scalar loss, averaged over the batch.
        """
        # Predictions are clipped away from exactly 0 or 1 so that
        # log(predictions) and log(1 - predictions) never evaluate
        # log(0), which would produce -inf and subsequently nan once
        # multiplied by a zero target or combined in backward().
        self.predictions = np.clip(predictions, 1e-12, 1 - 1e-12)
        self.targets = targets

        # Standard binary cross-entropy per example:
        #   -[t * log(a) + (1 - t) * log(1 - a)]
        # averaged over all m examples in the batch.
        per_example_loss = -(
            self.targets * np.log(self.predictions)
            + (1 - self.targets) * np.log(1 - self.predictions)
        )
        return float(np.mean(per_example_loss))

    def backward(self) -> np.ndarray:
        """Compute the gradient of the loss w.r.t. predictions.

        Returns:
            np.ndarray: dL/da, same shape as the predictions passed
            to forward. Uses the same clipped predictions here as in
            forward; see "Numerical stability" above.
        """
        # This is the start of the gradient chain, so there is no
        # grad_output argument to incorporate; the derivative is taken
        # directly with respect to the loss itself.
        #
        # For a single example, d/da [-(t*log(a) + (1-t)*log(1-a))]
        # equals (a - t) / (a * (1 - a)). The batch average in forward()
        # introduces a factor of 1/m here as well, where m is the
        # number of examples, so that this gradient is consistent with
        # the scalar loss returned by forward().
        m = self.targets.shape[0]
        grad_input = (self.predictions - self.targets) / (
            self.predictions * (1 - self.predictions)
        )
        return grad_input / m
