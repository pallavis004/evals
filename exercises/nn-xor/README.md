# Neural Network from Scratch

Intern onboarding task · Python

## 1. Objective

This task is your first contribution to the codebase. The scope is small on purpose. The focus is how you branch, commit, structure code, test it, and use your tools — not just whether the final model works. Treat this exactly like a real ticket.

## 2. Problem statement

Implement a small feedforward neural network in Python, from scratch, using only NumPy for the math. No PyTorch, TensorFlow, scikit-learn, Keras, or JAX. The network must learn the XOR function: given two binary inputs, predict the correct XOR output. XOR is not linearly separable, so a single-layer perceptron cannot solve it — your network needs at least one hidden layer.

### 2.1 Constraints

- Python 3.10 or later.
- NumPy is the only allowed dependency for the model itself. Testing libraries (pytest) are fine.
- Implement the forward pass, the loss function, and backpropagation yourself. No autograd.
- Network architecture: 2 input units, at least 1 hidden layer with a nonlinear activation (sigmoid, tanh, or ReLU), 1 output unit.
- Training must use gradient descent (plain or with momentum), implemented by you.

### 2.2 Reference interface

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 learning_rate: float = 0.1) -> None:
        ...

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Returns the network's prediction for input x."""
        ...

    def backward(self, x: np.ndarray, y: np.ndarray, output: np.ndarray) -> None:
        """Computes gradients and updates weights in place."""
        ...

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int) -> list[float]:
        """Trains the network, returns the loss recorded at each epoch."""
        ...
```

### 2.3 Data the network must learn

| X1 | X2 | Target |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

### 2.4 Deliverables

- A `NeuralNetwork` class (or equivalent) implementing forward pass, backward pass, and a training loop.
- A training script that trains on the XOR data and prints the final predictions.
- Unit tests (pytest) that check the trained network predicts the correct XOR output for all 4 input pairs, within a stated tolerance (for example, output rounds to the correct class).
- A short README section, or PR description, stating the chosen architecture (hidden layer size, activation function) and the final loss after training.
- A `requirements.txt` or equivalent listing NumPy and pytest.

### 2.5 Acceptance criteria

- Network correctly predicts all 4 XOR input pairs after training.
- No use of a deep learning framework for the model or training loop.
- Forward and backward pass are implemented as separate, readable functions or methods.
- Tests are present and pass.
- PR follows the branching and commit standards in [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

## 3. Submitting your solution

Follow the fork + PR workflow and standards described in the repo's top-level [`CONTRIBUTING.md`](../../CONTRIBUTING.md). Put your work in `solutions/<your-name>/nn-xor/` in your fork, then open a PR into `VitalView-AI/evals` `main`.
