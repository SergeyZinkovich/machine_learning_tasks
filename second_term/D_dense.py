import numpy as np
from typing import Optional, Tuple


class Dense:
    """Implements fully-connected layer"""

    __rng = np.random.default_rng()

    def __init__(self, n_in: int, n_out: int, use_bias: bool = True):
        """Initializes Dense layer.
        The weights are initialized using uniformly distributed values in range [-1, 1]. Bias vector is not initialized if `use_bias` is False.
        Weigths matrix has the shape (`n_in`, `n_out`), bias vector has the shape (`n_out`, ).

        Arguments:
            n_in: Positive integer, dimensionality of input space.
            n_out: Positive integer, dimensionality of output space.
            use_bias: Whether the layer uses a bias vector."""
        self.shape = (n_in, n_out)
        self.W = self.__rng.random(size=(n_in, n_out))
        self.b = self.__rng.random(size=(n_out))
        self.use_bias = use_bias

    @property
    def weights(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Returns weights used by the layer."""
        if self.use_bias:
            return self.W, self.b
        else:
            return (self.W, )

    @property
    def input(self) -> np.ndarray:
        """Returns the last input received by the layer"""
        return self.input_

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Performs the layer forward pass.

        Arguments:
            x: Input array of shape (`batch_size`, `n_in`)

        Returns:
            An array of shape (`batch_size`, `n_out`)"""
        self.input_ = x

        if self.use_bias:
            x = x @ self.W + self.b
        else:
            x = x @ self.W
        return x

    def grad(self, gradOutput: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Computes layer gradients

        Arguments:
            gradOutput: Gradient of loss function with respect to the layer output, an array of shape (`batch_size`, `n_out`).

        Returns:
            A tuple object:
                Gradient of loss function with respect to the layer input, an array of shape (`batch_size`, `n_in`)
                Gradient of loss function with respect to the layer's weights:
                    An array of shape (`n_in`, `n_out`).
                    Optional array of shape (`n_out`, )."""
        return (gradOutput @ self.W.T , (self.input_.T @ gradOutput, gradOutput.sum(axis=0)))