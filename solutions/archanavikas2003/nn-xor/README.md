# XOR Neural Network (NumPy, from scratch)

A minimal feedforward neural network that learns the XOR function. Forward
pass, loss, and backpropagation are all implemented by hand with NumPy — no
autograd and no deep learning framework (no PyTorch, TensorFlow, Keras, JAX,
or scikit-learn).

XOR is not linearly separable, so a single-layer perceptron cannot solve it.
The network therefore uses one hidden layer with a nonlinear activation.

## Architecture

| Component            | Choice                                  |
| -------------------- | --------------------------------------- |
| Input units          | 2                                       |
| Hidden layer         | 1 layer, **4 units**, **sigmoid**       |
| Output unit          | 1, **sigmoid** (read as a probability)  |
| Loss                 | Binary cross-entropy                    |
| Optimizer            | Full-batch gradient descent (hand-written) |
| Learning rate        | 1.0                                     |
| Epochs               | 10,000                                  |
| Weight init          | `numpy.random.default_rng(seed=0)`, normal(0, 1); biases zero |

Data flow: `2 -> sigmoid(4) -> sigmoid(1)`.

## Result

With the settings above (seed 0), training converges cleanly:

```
XOR predictions after training:
  [0 0] -> 0.0001 -> 0
  [0 1] -> 0.9987 -> 1
  [1 0] -> 0.9988 -> 1
  [1 1] -> 0.0020 -> 0

Final loss: 0.0013
```

**Final loss after training: ~0.0013** (binary cross-entropy). All four input
pairs are classified correctly, and each output is within 0.1 of its target —
so predictions are confident, not merely on the correct side of the 0.5
threshold.

## Files

- `nn.py` — the model and training loop; run it directly to train and print
  predictions.
- `requirements.txt` — dependencies (NumPy, pytest).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```





The suite trains the network once (shared via a fixture) and checks that:

- all four input pairs round to the correct XOR class,
- each pair is correct individually (parametrized, so a failure names the pair),
- every output is within a 0.1 tolerance of its target,
- the final training loss is below 0.01,
- results are reproducible for a fixed seed.