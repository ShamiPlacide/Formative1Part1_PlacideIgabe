# Formative 1: A Neural Network Library from Scratch

This project implements a small neural network library in NumPy. It is built chapter by chapter and includes a base module interface, layers, activations, losses, an optimizer, and a training loop.

```
Formative1Part1_PlacideIgabe/
├── nn/
│   ├── __init__.py
│   ├── module.py
│   ├── layers/
│   │   ├── __init__.py
│   │   └── linear.py
│   ├── activations/
│   │   ├── __init__.py
│   │   ├── relu.py
│   │   ├── sigmoid.py
│   │   └── softmax.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── cross_entropy_loss.py
│   │   └── categorical_cross_entropy_loss.py
│   └── optim/
│       ├── __init__.py
│       └── sgd.py
├── tests/
├── main.py
└── README.md
```

## Files

- **`nn/module.py`**: Defines the base `Module` class that all layers and activations inherit from, providing a shared interface for the forward pass, backward pass, and parameters.
- **`nn/layers/linear.py`**: Implements the fully connected layer, which applies a weighted sum and bias to its inputs.
- **`nn/activations/relu.py`**: Implements the ReLU activation, which sets negative values to zero.
- **`nn/activations/sigmoid.py`**: Implements the Sigmoid activation, which maps values to the range 0 to 1 for binary classification.
- **`nn/activations/softmax.py`**: Implements the Softmax activation, which converts scores into class probabilities for multi-class classification.
- **`nn/losses/cross_entropy_loss.py`**: Implements binary cross-entropy loss, used together with Sigmoid.
- **`nn/losses/categorical_cross_entropy_loss.py`**: Implements categorical cross-entropy loss, used together with Softmax.
- **`nn/optim/sgd.py`**: Implements stochastic gradient descent, which updates the model parameters using their gradients.
- **`main.py`**: Trains a small network on a toy dataset and reports the resulting loss and accuracy.
- **`tests/`**: Contains the provided test suite used to verify each chapter. These files should not be edited.

## Reference
- Consulted NumPy documentation for np.random.uniform and np.clip.
- Used Claude to check conceptual understanding of the Module base class contract and shape reasoning for activation functions.
