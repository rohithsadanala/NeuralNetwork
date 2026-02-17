import numpy as np
from sklearn.datasets import fetch_openml
import sys

def load_and_prep_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X_raw = mnist["data"]
    y_raw = mnist["target"].astype(int)

    X_raw = X_raw / 255.0

    num_classes = 10
    y_encoded = np.eye(num_classes)[y_raw]

    split_idx = 60000
    X_train, X_test = X_raw[:split_idx], X_raw[split_idx:]
    y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]

    return X_train, y_train, X_test, y_test


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.params = {
            'W1': np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size),
            'b1': np.zeros((1, hidden_size)),
            'W2': np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size),
            'b2': np.zeros((1, output_size))
        }

    def relu(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        Z1 = np.dot(X, self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)

        Z2 = np.dot(A1, self.params['W2']) + self.params['b2']
        A2 = self.softmax(Z2)

        self.cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
        return A2

    def backward(self, y_true):
        m = y_true.shape[0]
        X = self.cache['X']
        A1 = self.cache['A1']
        A2 = self.cache['A2']
        W2 = self.params['W2']

        dZ2 = A2 - y_true
        dW2 = (1 / m) * np.dot(A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * (self.cache['Z1'] > 0)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        self.params['W1'] -= self.learning_rate * dW1
        self.params['b1'] -= self.learning_rate * db1
        self.params['W2'] -= self.learning_rate * dW2
        self.params['b2'] -= self.learning_rate * db2

    def train(self, X_train, y_train, epochs=10, batch_size=64):
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                self.forward(X_batch)
                self.backward(y_batch)

            if (epoch + 1) % 1 == 0:
                accuracy = self.evaluate(X_train, y_train)
                print(f"Epoch {epoch + 1}/{epochs} - Training Accuracy: {accuracy:.2f}%")

    def predict(self, X):
        A2 = self.forward(X)
        return np.argmax(A2, axis=1)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels) * 100
        return accuracy

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_and_prep_data()

    nn = NeuralNetwork(input_size=784, hidden_size=128, output_size=10, learning_rate=0.1)

    print("\nStarting Training...")
    nn.train(X_train, y_train, epochs=10, batch_size=64)

    print("\nEvaluating on Test Data...")
    test_accuracy = nn.evaluate(X_test, y_test)
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")