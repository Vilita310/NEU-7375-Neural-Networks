import numpy as np

# Activation function classes
class Activation:
    def __init__(self, activation_type='linear'):
        self.activation_type = activation_type

    def __call__(self, z):
        if self.activation_type == 'linear':
            return self.linear(z)
        elif self.activation_type == 'relu':
            return self.relu(z)
        elif self.activation_type == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation_type == 'tanh':
            return self.tanh(z)
        elif self.activation_type == 'softmax':
            return self.softmax(z)

    def derivative(self, z):
        if self.activation_type == 'linear':
            return np.ones_like(z)
        elif self.activation_type == 'relu':
            return (z > 0).astype(float)
        elif self.activation_type == 'sigmoid':
            s = self.sigmoid(z)
            return s * (1 - s)
        elif self.activation_type == 'tanh':
            t = self.tanh(z)
            return 1 - t**2
        elif self.activation_type == 'softmax':
            s = self.softmax(z)
            return s * (1 - s)

    def linear(self, z):
        return z

    def relu(self, z):
        return np.maximum(0, z)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def tanh(self, z):
        return np.tanh(z)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Shift for numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# Loss function class (using cross-entropy)
class Loss:
    def __init__(self, loss_type='cross_entropy'):
        self.loss_type = loss_type

    def compute(self, y_true, y_pred):
        if self.loss_type == 'cross_entropy':
            return self.cross_entropy(y_true, y_pred)

    def cross_entropy(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred)) / m

    def gradient(self, y_true, y_pred):
        return y_pred - y_true


# Deep Neural Network class
class DeepNeuralNetwork:
    def __init__(self, layers, activations):
        self.layers = layers  # List of neurons in each layer
        self.activations = [Activation(activation) for activation in activations]
        self.parameters = self._initialize_parameters()

    def _initialize_parameters(self):
        parameters = {}
        np.random.seed(42)

        for l in range(1, len(self.layers)):
            parameters[f"W{l}"] = np.random.randn(self.layers[l], self.layers[l-1]) * 0.01
            parameters[f"b{l}"] = np.zeros((self.layers[l], 1))

        return parameters

    def forward(self, X):
        cache = {"A0": X}

        for l in range(1, len(self.layers)):
            W = self.parameters[f"W{l}"]
            b = self.parameters[f"b{l}"]

            Z = np.dot(W, cache[f"A{l-1}"]) + b
            A = self.activations[l-1](Z)

            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A

        return cache[f"A{len(self.layers) - 1}"], cache

    def compute_cost(self, y_true, y_pred):
        loss = Loss(loss_type='cross_entropy')
        return loss.compute(y_true, y_pred)

    def backward(self, X, y_true, cache):
        grads = {}
        m = X.shape[1]
        L = len(self.layers) - 1

        # Calculate the gradient for the output layer
        A_final = cache[f"A{L}"]
        dA = A_final - y_true
        grads[f"dA{L}"] = dA

        # Backpropagate through layers
        for l in reversed(range(1, L+1)):
            dA = grads[f"dA{l}"]
            Z = cache[f"Z{l}"]
            A_prev = cache[f"A{l-1}"]
            W = self.parameters[f"W{l}"]

            # Activation gradient
            dZ = dA * self.activations[l-1].derivative(Z)
            grads[f"dW{l}"] = np.dot(dZ, A_prev.T) / m
            grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True) / m
            grads[f"dA{l-1}"] = np.dot(W.T, dZ)

        return grads

    def update_parameters(self, grads, learning_rate):
        for l in range(1, len(self.layers)):
            self.parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
            self.parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            y_pred, cache = self.forward(X)
            cost = self.compute_cost(y, y_pred)
            grads = self.backward(X, y, cache)
            self.update_parameters(grads, learning_rate)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.4f}")

    def predict(self, X):
        y_pred, _ = self.forward(X)
        return np.argmax(y_pred, axis=0)


# Example usage
if __name__ == "__main__":
    # Sample data (random for demonstration purposes)
    X = np.random.randn(784, 1000)  # Input data: 784 features (e.g., flattened 28x28 images), 1000 examples
    y = np.random.randint(0, 10, size=(1, 1000))  # Labels: 10 classes
    y_one_hot = np.eye(10)[y.flatten()].T  # One-hot encoded labels

    # Define network architecture
    layers = [784, 64, 10]  # Input layer (784), hidden layer (64 neurons), output layer (10 neurons)
    activations = ['relu', 'softmax']  # ReLU for hidden layer, Softmax for output layer

    # Initialize model
    model = DeepNeuralNetwork(layers, activations)

    # Train model
    model.train(X, y_one_hot, epochs=1000, learning_rate=0.01)

    # Predict on new data
    predictions = model.predict(X)
    print("Predictions:", predictions)
