# Handwritten Digit Classifier from Scratch (MNIST)

Intern onboarding task · Python · Level 2

## 1. Objective

This is the natural next step after [`nn-xor`](../nn-xor/README.md): move from a hand-built 2-input binary classifier to a small multi-class classifier on real data. It adds several new concepts at once without being a huge leap, and it re-tests whether you actually understood the XOR gradient derivation or just memorized the specific line.

## 2. Background: from two classes to ten

The XOR task asked "is the output 0 or 1?" — a single number between 0 and 1 (via sigmoid) was enough to answer that. Digit classification asks a different kind of question: "which one of these 10 digits is this?" A single number can't represent a choice among 10 options, so the output layer needs 10 units, one per class — and you need a way to turn 10 raw numbers into something that behaves like a probability distribution over those 10 classes.

That's what **softmax** does: it exponentiates every raw score and divides each by the sum of all the exponentials, so the 10 outputs are all positive and sum to exactly 1. Why exponentiate instead of just normalizing the raw scores directly? Because raw scores can be negative, and because exponentiation exaggerates the gap between the largest score and the rest — a network that's genuinely confident should produce a sharply peaked distribution, not a nearly-uniform one.

The matching loss function is **categorical cross-entropy**: instead of comparing one predicted probability to one target like binary cross-entropy did, it compares the predicted probability assigned to the *correct* class to 1, and penalizes the network in proportion to how far off that probability was. Remarkably, when you work through the calculus, the gradient of softmax combined with categorical cross-entropy collapses to `predictions - one_hot_labels` — the exact same shape as the XOR task's `output - target`, just with 10 numbers instead of 1. That's not a coincidence; it's the same underlying pattern generalizing to more classes. If you can't immediately see why, that's a sign to go back and re-derive the XOR gradient rather than move on.

The other new ingredient is scale. Four rows of data can be shown to the network all at once, every step (full-batch gradient descent). Thousands of images can't be handled that way efficiently — training would be slow, and early on, the averaged gradient over the whole dataset is a less useful signal than it sounds. **Mini-batch SGD** shows the network a random subset of the data at a time, updates weights after each subset, and reshuffles the data every epoch — trading a noisier per-step gradient estimate for many more update steps per pass through the data. Combined with a real weight initialization strategy (naive small-random weights tend to make every hidden unit start out too similar to its neighbors, or push activations into the flat, near-zero-gradient regions of sigmoid/tanh), this is what makes training a deeper network on real data actually converge instead of stalling.

## 3. What's new versus the XOR task

| XOR (level 1) | Digit classifier (level 2) |
|---|---|
| 2 inputs | 784 inputs (28x28 pixels) |
| 1 hidden layer | 2 hidden layers |
| Binary output (sigmoid) | 10-class output (softmax) |
| Binary cross-entropy | Categorical cross-entropy |
| Full-batch gradient descent (4 rows) | Mini-batch SGD (thousands of rows) |
| No train/test split | Train/test split, must report test accuracy |
| No weight init strategy needed | Requires proper init (Xavier/He) or it won't converge |

## 4. Problem statement

Implement a small feedforward neural network in Python, from scratch, using only NumPy for the math. No PyTorch, TensorFlow, scikit-learn (except for loading the dataset — see below), Keras, or JAX. The network must classify handwritten digit images into 10 classes (0-9).

### 4.1 Constraints

- Python 3.10 or later.
- NumPy is the only allowed dependency for the model itself. Testing libraries (pytest) are fine.
- Implement the forward pass, the loss function, and backpropagation yourself. No autograd.
- Network architecture: input layer sized to the flattened image, 2 hidden layers with a nonlinear activation (sigmoid, tanh, or ReLU), 10-unit softmax output layer.
- Loss: categorical cross-entropy.
- Training must use **mini-batch** stochastic gradient descent, implemented by you: shuffle the training data each epoch, split into batches, iterate. Do not rely on a framework's `DataLoader` or batching utility.
- Weights must use a real initialization strategy (Xavier/He, or equivalent reasoning) — naive small-random or zero init will not converge reliably at this size and that's the point.
- Load MNIST from a CSV, or use `sklearn.datasets.load_digits` (8x8 images, 10 classes) if full MNIST is too heavy for your setup. Either is fine — **state which one you used** in your PR description.

### 4.2 Reference interface

```python
import numpy as np

class DigitClassifier:
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int,
                 learning_rate: float = 0.1) -> None:
        ...

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Returns class probabilities (softmax output) for a batch X."""
        ...

    def backward(self, X: np.ndarray, y_one_hot: np.ndarray, output: np.ndarray) -> None:
        """Computes gradients and updates weights in place."""
        ...

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int,
              batch_size: int = 32) -> list[float]:
        """Trains with manual mini-batching (shuffle + split each epoch).
        Returns the loss recorded at each epoch."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted class labels for a batch X."""
        ...

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Returns classification accuracy on the given data."""
        ...
```

### 4.3 Deliverables

Everything from [`nn-xor`](../nn-xor/README.md)'s deliverables, plus:

- Train/test split logic (implemented by you, or via a standard splitting utility — your choice, but state it).
- A manual mini-batch training loop (shuffle each epoch, split into batches).
- Train **and** test accuracy reported separately in your PR description or README section. This catches overfitting and forces you to reason about generalization instead of just watching loss go down.
- A confusion matrix or per-class accuracy breakdown, not just an overall accuracy number — this stops a broken class from hiding behind a decent average.
- A training script that trains the model and prints train/test accuracy and the per-class breakdown.
- Unit tests (pytest) covering at minimum: softmax output sums to 1 per row, forward pass shape correctness, and a trained-model accuracy threshold on the test set.
- A `requirements.txt` or equivalent (NumPy, pytest, and `scikit-learn` only if used for data loading or the train/test split).

### 4.4 Acceptance criteria

- Model trains via manually-implemented mini-batch SGD (no framework `DataLoader`/batching helper).
- Softmax + categorical cross-entropy implemented from scratch, no autograd.
- Test accuracy reported separately from train accuracy, and is meaningfully above chance (>> 10%).
- Per-class accuracy or confusion matrix is present and reviewed in the PR description — call out any class that underperforms and why.
- No use of a deep learning framework for the model or training loop.
- Tests are present and pass.
- PR follows the branching and commit standards in [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

## 5. Verification questions

Be ready to answer these in review — they're the actual point of the exercise:

1. Why does softmax use `exp` and normalize, instead of just picking the largest raw score?
2. Derive why `predictions - one_hot_labels` is the correct gradient for softmax + cross-entropy — same idea as the XOR task's sigmoid gradient, but now with 10 outputs instead of 1.
3. Your test accuracy is X%, your train accuracy is Y%. If Y is much higher than X, what's happening, and what's one thing in your code that would fix it?
4. Why does weight initialization matter more here than it did for the 2-input XOR network?

## 6. Submitting your solution

Follow the fork + PR workflow and standards described in the repo's top-level [`CONTRIBUTING.md`](../../CONTRIBUTING.md). Put your work in `solutions/<your-name>/mnist-classifier/` in your fork, then open a PR into `VitalView-AI/evals` `main`.
