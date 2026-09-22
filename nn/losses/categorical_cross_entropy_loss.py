"""Categorical cross-entropy loss, for one-hot multi-class targets."""
import numpy as np


class CategoricalCrossEntropyLoss:
    """Categorical cross-entropy loss over C classes.

    Does not subclass Module; see the note in Chapter 6.
    """

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute the average categorical cross-entropy loss.

        Args:
            predictions (np.ndarray): softmax probabilities,
                shape (m, C). Clip away from exactly 0 before use;
                see "The same clipping requirement as Chapter 6" above.
            targets (np.ndarray): one-hot true labels, shape (m, C).

        Returns:
            float: the scalar loss, averaged over the batch.
        """
        # Predictions are clipped away from exactly 0 so that
        # log(predictions) never evaluates log(0), which would produce
        # -inf and subsequently nan once combined with the target in
        # backward(). Only the lower bound matters here, since a
        # softmax output can equal 1 without causing any issue for
        # log(), but clipping symmetrically keeps this consistent with
        # the binary case in Chapter 6.
        self.predictions = np.clip(predictions, 1e-12, 1 - 1e-12)
        self.targets = targets

        # For one-hot targets, only the log-probability of the correct
        # class contributes to each example's loss, since targets is 0
        # everywhere else in that row. Summing over classes and then
        # averaging over the batch gives the standard categorical
        # cross-entropy:
        #   -mean_over_m( sum_over_c( t_c * log(a_c) ) )
        per_example_loss = -np.sum(self.targets * np.log(self.predictions), axis=1)
        return float(np.mean(per_example_loss))

    def backward(self) -> np.ndarray:
        """Compute the gradient of the loss w.r.t. predictions.

        Returns:
            np.ndarray: dL/da, shape (m, C), same shape as the
            predictions passed to forward. Uses the same clipped
            predictions here as in forward.
        """
        # This is the start of the gradient chain, so there is no
        # grad_output argument to incorporate.
        #
        # For a single example, d/da_c [-sum_c(t_c * log(a_c))] equals
        # -t_c / a_c, elementwise across classes. The batch average in
        # forward() introduces a factor of 1/m here as well, where m
        # is the number of examples, so that this gradient is
        # consistent with the scalar loss returned by forward().
        #
        # Note that this gradient is taken directly with respect to
        # the softmax probabilities, not fused with the Softmax
        # layer's own backward(). If predictions come from a Softmax
        # layer, its backward() is still called afterward, in the
        # usual chain, rather than substituted with the simplified
        # (a - t) form that only holds for the fused combination.
        m = self.targets.shape[0]
        grad_input = -self.targets / self.predictions
        return grad_input / m
