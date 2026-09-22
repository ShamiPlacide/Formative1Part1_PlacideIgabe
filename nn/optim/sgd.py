"""Stochastic gradient descent optimizer."""
import numpy as np


class SGD:
    """Vanilla SGD: param -= lr * grad, for every tracked parameter."""

    def __init__(
        self,
        parameters: list[tuple[np.ndarray, np.ndarray]],
        lr: float,
    ) -> None:
        """Store the parameters to update and the learning rate.

        Args:
            parameters (list[tuple[np.ndarray, np.ndarray]]):
                (param, grad) pairs, as returned by a module's
                parameters() method.
            lr (float): learning rate, the step size eta.
        """
        # The (param, grad) pairs are stored as given, without
        # copying the underlying arrays. Since numpy arrays are
        # mutable objects, keeping these references allows step() and
        # zero_grad() to modify the same arrays that the model's
        # modules hold, rather than a detached copy.
        self.parameters = parameters
        self.lr = lr

    def step(self) -> None:
        """Apply one update step to every tracked parameter.

        Updates every param in place using its current grad,
        following param -= lr * grad. Does not return anything.
        """
        # Each param is updated using in-place subtraction, so that
        # the same array object referenced by the owning module is
        # modified directly, rather than a new array being created
        # and rebound only to a local name here.
        for param, grad in self.parameters:
            param -= self.lr * grad

    def zero_grad(self) -> None:
        """Reset every tracked parameter's gradient to zero.

        Does not return anything.
        """
        # As in step(), each grad array is cleared in place using
        # ellipsis assignment, so that the owning module's stored
        # gradient array is actually reset, rather than reassigning
        # the local name grad to a freshly created zero array.
        for param, grad in self.parameters:
            grad[...] = 0
