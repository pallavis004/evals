import numpy as np
def sigmoid (z):
    return 1 / (1 + np.exp(-z))
def forward(x, w1, b1, w2, b2):
    a1 = sigmoid(x @ w1 + b1)
    a2 = sigmoid(a1 @ w2 + b2)
    return a1, a2
def backward(x, y, a1, a2, w2):
    dz2 = (a2-y)/len(x)
    dw2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    dz1 = dz2 @ w2.T * a1 * (1-a1)
    dw1 = x.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)
    return dw1, db1, dw2, db2

def loss(a2, y):
    """Binary cross-entropy between the output a2 and the targets y."""
    p = np.clip(a2, 1e-12, 1 - 1e-12)   # keep log() from blowing up at 0 or 1
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


x = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

rng = np.random.default_rng(0)
w1 = rng.normal(0,1,size=(2,4))
b1 = np.zeros((1,4))
w2 = rng.normal(0,1,size=(4,1))
b2 = np.zeros((1,1))

lr = 1.0

for epoch in range(10000):
    a1, a2 = forward(x, w1, b1, w2, b2)
    if epoch%1000 == 0:
        print(f"epoch {epoch:5d}  loss {loss(a2, y):.4f}")
    dw1, db1, dw2, db2 = backward(x, y, a1, a2, w2)
    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2



_, out = forward(x, w1, b1, w2, b2)
for xi, o in zip(x, out.ravel()):
    print(f"{xi.astype(int)} -> {o:.4f} -> {int(o >= 0.5)}")
