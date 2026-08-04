# Neural Network from Scratch

Intern onboarding task · Python

## 1. Objective

This task is your first contribution to the codebase. The scope is small on purpose. The focus is how you branch, commit, structure code, test it, and use your tools — not just whether the final model works. Treat this exactly like a real ticket.

## 2. Background: what a neural network is doing here

At its core, a neural network is a chain of two operations repeated a few times: a **weighted sum** of the inputs (a linear transformation), followed by a **nonlinear activation function** applied to that sum. Stack a few of these together and the network can approximate much more complex functions than either operation could alone.

A single layer with no hidden units — just weighted inputs feeding straight into an output — can only draw a straight line (or, in higher dimensions, a flat plane) to separate its predictions into two classes. That's fine for problems where the two classes really are separable by a straight line. XOR isn't one of them: plot the four input pairs on a 2D grid and color them by their target — `(0,0)` and `(1,1)` are one class, `(0,1)` and `(1,0)` are the other. There is no single straight line that puts both same-colored points on the same side. This is the textbook example of a **non-linearly separable** problem, and it's exactly why XOR defeated the original single-layer perceptron in the 1960s and briefly stalled neural network research.

A **hidden layer** fixes this by first transforming the input into a new coordinate space — one where the problem *becomes* linearly separable — and only then drawing the straight line. The hidden layer's job during training is to discover a transformation that does this; nothing about it is hardcoded, it's learned entirely from the gradient signal.

That gradient signal comes from **backpropagation**: after a forward pass produces a prediction and you compute how wrong it was (the loss), backprop uses the chain rule to work backward through the network, computing how much each individual weight contributed to that error. **Gradient descent** then nudges every weight a small step in the direction that reduces the error. Repeat that thousands of times and the network's weights converge on a transformation that solves the problem.

One thing to watch for: when you derive the gradient for a sigmoid output with binary cross-entropy loss, it collapses to something deceptively simple — `output - target`. It's easy to copy that line from a reference without ever doing the chain rule that produces it. Do the derivation by hand at least once. The next exercise in this series builds directly on this same gradient, with more outputs, and assumes you actually understand where it comes from.

## 3. Problem statement

Implement a small feedforward neural network in Python, from scratch, using only NumPy for the math. No PyTorch, TensorFlow, scikit-learn, Keras, or JAX. The network must learn the XOR function: given two binary inputs, predict the correct XOR output. XOR is not linearly separable, so a single-layer perceptron cannot solve it — your network needs at least one hidden layer.

### 3.1 Constraints

- Python 3.10 or later.
- NumPy is the only allowed dependency for the model itself. Testing libraries (pytest) are fine.
- Implement the forward pass, the loss function, and backpropagation yourself. No autograd.
- Network architecture: 2 input units, at least 1 hidden layer with a nonlinear activation (sigmoid, tanh, or ReLU), 1 output unit.
- Training must use gradient descent (plain or with momentum), implemented by you.

### 3.2 Reference interface

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

### 3.3 Data the network must learn

| X1 | X2 | Target |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

### 3.4 Deliverables

- A `NeuralNetwork` class (or equivalent) implementing forward pass, backward pass, and a training loop.
- A training script that trains on the XOR data and prints the final predictions.
- Unit tests (pytest) that check the trained network predicts the correct XOR output for all 4 input pairs, within a stated tolerance (for example, output rounds to the correct class).
- A short README section, or PR description, stating the chosen architecture (hidden layer size, activation function) and the final loss after training.
- A `requirements.txt` or equivalent listing NumPy and pytest.

### 3.5 Acceptance criteria

- Network correctly predicts all 4 XOR input pairs after training.
- No use of a deep learning framework for the model or training loop.
- Forward and backward pass are implemented as separate, readable functions or methods.
- Tests are present and pass.
- PR follows the branching and commit standards in [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

## 4. Submitting your solution

Follow the fork + PR workflow and standards described in the repo's top-level [`CONTRIBUTING.md`](../../CONTRIBUTING.md). Put your work in `solutions/<your-name>/nn-xor/` in your fork, then open a PR into `VitalView-AI/evals` `main`.
