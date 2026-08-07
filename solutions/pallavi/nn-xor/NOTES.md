# XOR Neural Network — Architecture Notes

## Architecture

- Input units: 2
- Hidden layer: 1 layer, 4 units, tanh activation
- Output unit: 1, sigmoid activation (read as a probability)
- Loss: Binary cross-entropy
- Optimizer: Full-batch gradient descent (hand-written), learning rate 1
- Epochs: 10,000
- Weight init: `np.random.randn`, biases zero, seed 0

## Why tanh for the hidden layer?

Tanh is zero-centered (-1 to 1), which generally makes gradient descent
converge faster than sigmoid, which is always positive. Sigmoid is kept
for the output layer since XOR is a binary classification problem and
a 0-1 output can be read directly as a probability.

## Result

Final loss after training: 0.0002

| Input | Predicted | Expected |
|---|---|---|
| [0, 0] | 0.0002 -> 0 | 0 |
| [0, 1] | 0.9997 -> 1 | 1 |
| [1, 0] | 1.0000 -> 1 | 1 |
| [1, 1] | 0.0003 -> 0 | 0 |

All 4 XOR pairs predicted correctly.