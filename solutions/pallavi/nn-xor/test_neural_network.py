import numpy as np
from neural_network import forward, backward, calculate_loss


def train_xor_network():
    """Train a fresh XOR network and return the inputs, targets, and final predictions."""
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
        dW1, dB1, dW2, dB2 = backward(X, Y, hidden, output, W2)
        W1 = W1 - learning_rate * dW1
        B1 = B1 - learning_rate * dB1
        W2 = W2 - learning_rate * dW2
        B2 = B2 - learning_rate * dB2

    _, final_output = forward(X, W1, B1, W2, B2)
    return X, Y, final_output


def test_xor_predictions_are_correct():
    """Check that the trained network's rounded predictions match the XOR truth table."""
    _, Y, output = train_xor_network()
    predicted_classes = (output >= 0.5).astype(int)
    expected_classes = Y.astype(int)
    assert np.array_equal(predicted_classes, expected_classes)


def test_loss_is_low_after_training():
    """Check that the trained network's final loss is below an acceptable threshold."""
    _, Y, output = train_xor_network()
    final_loss = calculate_loss(output, Y)
    assert final_loss < 0.01