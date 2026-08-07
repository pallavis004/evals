import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(X, W1, B1, W2, B2):
    hidden = np.tanh(np.dot(X, W1) + B1)
    output = sigmoid(np.dot(hidden, W2) + B2)
    return hidden, output


def backward(X, Y, hidden, output, W2):
    m = X.shape[0]
    output_error = (output - Y) / m
    dW2 = np.dot(hidden.T, output_error)
    dB2 = np.sum(output_error, axis=0, keepdims=True)

    hidden_error = np.dot(output_error, W2.T) * (1 - hidden ** 2)
    dW1 = np.dot(X.T, hidden_error)
    dB1 = np.sum(hidden_error, axis=0, keepdims=True)

    return dW1, dB1, dW2, dB2


def calculate_loss(prediction, target):
    prediction = np.clip(prediction, 1e-12, 1 - 1e-12)
    return -np.mean(target * np.log(prediction) +
                    (1 - target) * np.log(1 - prediction))