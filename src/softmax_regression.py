import numpy as np

def softmax(logits: np.ndarray) -> np.ndarray:
    # Subtraction of max for numerical stability
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    n = y.shape[0]
    # Add epsilon to prevent log(0)
    return -np.mean(np.log(probs[np.arange(n), y] + 1e-12))

def train_softmax(X, y, lr=1e-1, epochs=200, reg=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    k = int(np.max(y)) + 1
    
    # Initialize weights and biases
    W = 0.01 * rng.standard_normal((d, k))
    b = np.zeros((1, k))

    for _ in range(epochs):
        # Forward pass
        logits = X @ W + b
        probs = softmax(logits)
        loss = cross_entropy(probs, y) + 0.5 * reg * np.sum(W * W)

        # Backward pass (vectorized gradients)
        ds = probs.copy()
        ds[np.arange(n), y] -= 1
        ds /= n
        
        dW = X.T @ ds + reg * W
        db = np.sum(ds, axis=0, keepdims=True)

        # Gradient Descent update
        W -= lr * dW
        b -= lr * db

    return W, b, float(loss)

def predict(X, W, b):
    return np.argmax(X @ W + b, axis=1)

if __name__ == "__main__":
    print("Softmax Regression module initialized.")
