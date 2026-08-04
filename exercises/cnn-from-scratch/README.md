# Convolutional Neural Network from Scratch

Intern onboarding task · Python · Level 3

## 1. Objective

This is the third exercise in the from-scratch series, building directly on [`mnist-classifier`](../mnist-classifier/README.md). The scope, as always, is small on purpose — the focus is whether you understand *why* convolution works, not just whether you can copy a formula for it.

## 2. Background: why convolution, not just more dense layers

Your level-2 classifier flattened every image into a flat list of numbers before looking at it. That works, but it throws away something important: pixel `(5, 5)` being next to pixel `(5, 6)` meant nothing to that network — they were just two unrelated entries in a long vector. A dense layer learns a *separate* weight for every single pixel position, so if a "7" is drawn two pixels to the left of where it was in training, the network essentially has to relearn it from scratch in that new position.

Convolution fixes this with one idea: **use the same small set of weights (a kernel) everywhere in the image**, sliding it across every position and computing a local dot product at each one. A 3x3 edge-detecting kernel that works in the top-left corner of the image works exactly the same way in the bottom-right, because it's the *same weights*, reused, not a new set learned per position. This is called **weight sharing**, and it buys you two things at once: far fewer parameters (one kernel instead of one weight per pixel), and a network that responds to a pattern *wherever* it appears in the image, not just where it happened to appear in training.

The output of sliding a kernel across an image is called a **feature map** — at every position, it's a number saying how strongly that local patch matched what the kernel detects. Stack several kernels and you get several feature maps, each tuned to detect something different (an edge, a curve, a corner) — the network learns what those kernels should be, the same way it learned dense-layer weights in level 2.

**Pooling** (typically max pooling) then shrinks each feature map by keeping only the strongest response in each small region. This reduces computation for later layers, and adds a small amount of tolerance to exactly where a feature appeared, not just whether it appeared at all.

This is also directly why the real segmentation model in production (`L2 SEGMENT`, nnU-Net) is built almost entirely out of convolutions: it needs to recognize anatomical structures regardless of exactly where they sit in a scan, and it needs to do that efficiently across a full 3D volume. What you're building here, at toy scale, is the same core mechanism.

## 3. What's new versus the digit classifier (level 2)

| MNIST dense classifier (level 2) | CNN (level 3) |
|---|---|
| Flattened 1D input, no spatial meaning | 2D image input, spatial structure preserved |
| One weight per input pixel per hidden unit | Weight sharing — one small kernel reused across the whole image |
| No translation tolerance | Approximate translation invariance via conv + pooling |
| Simple matrix-multiply backward pass | Backward pass through convolution and pooling |
| Large first-layer parameter count | Far fewer parameters in the conv layers, for the same input |

## 4. Problem statement

Implement a small convolutional neural network in Python, from scratch, using only NumPy. It must classify the same digit dataset used in [`mnist-classifier`](../mnist-classifier/README.md) — use the same data source you used there (state which: full MNIST or `sklearn.datasets.load_digits`) so your results are directly comparable to your level-2 model.

### 4.1 Constraints

- Python 3.10 or later.
- NumPy is the only allowed dependency for the model itself. Testing libraries (pytest) are fine.
- Implement `conv2d` forward and backward yourself (loop-based or im2col — your choice, state which). No `scipy.signal.convolve`, no framework conv layers, no autograd.
- Implement `maxpool` forward and backward yourself.
- Architecture: at least 1 convolutional layer with pooling, followed by dense layer(s) reaching a 10-class softmax output. Reuse your level-2 softmax + categorical cross-entropy implementation for the output layer.
- Training must use mini-batch SGD, same as level 2 (manual shuffle + batching each epoch, no framework `DataLoader`).
- Use the same dataset and the same train/test split methodology as your level-2 solution, so the comparison in section 4.3 is apples-to-apples.

### 4.2 Reference interface

```python
import numpy as np

class ConvNet:
    def __init__(self, input_shape: tuple[int, int], num_filters: int,
                 kernel_size: int, hidden_size: int, output_size: int,
                 learning_rate: float = 0.1) -> None:
        ...

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Returns class probabilities (softmax output) for a batch X
        of shape (batch_size, height, width)."""
        ...

    def backward(self, X: np.ndarray, y_one_hot: np.ndarray, output: np.ndarray) -> None:
        """Computes gradients through the dense layers, pooling, and
        convolution, and updates weights in place."""
        ...

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int,
              batch_size: int = 32) -> list[float]:
        """Trains with manual mini-batching. Returns the loss recorded at
        each epoch."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted class labels for a batch X."""
        ...

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Returns classification accuracy on the given data."""
        ...
```

### 4.3 Deliverables

Everything from [`mnist-classifier`](../mnist-classifier/README.md)'s deliverables, plus:

- A hand-implemented `conv2d` (forward + backward) and `maxpool` (forward + backward).
- A **comparison table** in your PR description, level-2 model vs level-3 model, side by side:
  - Total trainable parameter count
  - Train accuracy
  - Test accuracy
  - Wall-clock training time (same number of epochs, same machine)
- A short written note on what you actually observed: did the CNN match, beat, or underperform the dense model on accuracy? Was it faster or slower to train, and does that match what you'd expect given the parameter counts? If something surprised you, say what and your best guess why.

### 4.4 Acceptance criteria

- `conv2d` and `maxpool` forward/backward are implemented from scratch — no framework or `scipy` convolution.
- Model trains via manually-implemented mini-batch SGD.
- Test accuracy is reported separately from train accuracy and is meaningfully above chance.
- The level-2 vs level-3 comparison table is present and discussed, not just pasted in.
- Tests are present and pass.
- PR follows the branching and commit standards in [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

## 5. Verification questions

1. Why does weight sharing (the same kernel reused across the image) make sense for image data but not for the level-2 dense layer's input?
2. Derive `dL/dkernel` for a single conv layer — how is it different in shape and computation from a dense layer's `dL/dW`?
3. You chose max pooling or average pooling — what's the tradeoff, and why did you pick the one you did?
4. Your CNN almost certainly has far fewer parameters than your level-2 dense model. Does it train faster or slower in wall-clock time, and why might that not match your intuition about "fewer parameters"?
5. If your CNN's accuracy is roughly the same as your dense model's on this small dataset, what did you actually gain by building it — and why does that gain matter more at the scale of a real image (e.g. a full CT slice) than it does for an 8x8 or 28x28 digit?

## 6. Submitting your solution

Follow the fork + PR workflow and standards described in the repo's top-level [`CONTRIBUTING.md`](../../CONTRIBUTING.md). Put your work in `solutions/<your-name>/cnn-from-scratch/` in your fork, then open a PR into `VitalView-AI/evals` `main`.
