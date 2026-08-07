"""Train a small feedforward network on the XOR dataset and print predictions."""
import numpy as np
from neural_network import forward, backward, calculate_loss

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

Y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

np.random.seed(0)
W1 = np.random.randn(2, 4)
B1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1)
B2 = np.zeros((1, 1))

learning_rate = 1
epochs = 10000

for epoch in range(epochs):
    hidden, output = forward(X, W1, B1, W2, B2)
    if epoch % 1000 == 0:
        print("Epoch:", epoch, "Loss:", round(calculate_loss(output, Y), 4))

    dW1, dB1, dW2, dB2 = backward(X, Y, hidden, output, W2)
    W1 = W1 - learning_rate * dW1
    B1 = B1 - learning_rate * dB1
    W2 = W2 - learning_rate * dW2
    B2 = B2 - learning_rate * dB2

_, prediction = forward(X, W1, B1, W2, B2)
print("\nFinal Predictions")
for inp, pred in zip(X, prediction):
    print(inp.astype(int), "->", round(pred[0], 4), "->", int(pred[0] >= 0.5))