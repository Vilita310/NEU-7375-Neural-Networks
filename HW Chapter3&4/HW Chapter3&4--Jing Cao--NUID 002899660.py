import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load and resize MNIST to 20x20 grayscale images
def prepare_data():
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist["data"], mnist["target"]
    X = X.to_numpy().reshape(-1, 28, 28)
    y = y.to_numpy().astype(int)
    
    # Resize to 20x20 images
    X_resized = np.array([cv2.resize(img, (20, 20)) for img in X])
    
    # Flatten each 20x20 image into a vector
    X_flat = X_resized.reshape(X_resized.shape[0], -1)
    
    # Normalize pixel values to [0, 1]
    X_flat = X_flat / 255.0
    
    return X_flat, y

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of sigmoid function
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Perceptron model class
class Perceptron:
    def __init__(self, input_size, output_size):
        # Initialize weights and biases
        self.W = np.random.randn(input_size, output_size) * 0.01  # Transmission matrix
        self.b = np.zeros((1, output_size))  # Bias
    
    def forward(self, X):
        # Forward propagation
        z = np.dot(X, self.W) + self.b
        a = sigmoid(z)
        return a
    
    def compute_cost(self, A, Y):
        # Compute cross-entropy loss
        m = Y.shape[0]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
        return np.squeeze(cost)
    
    def backward(self, X, A, Y):
        # Backpropagation to compute gradients
        m = X.shape[0]
        dz = A - Y
        dW = np.dot(X.T, dz) / m
        db = np.sum(dz) / m
        return dW, db
    
    def update_parameters(self, dW, db, learning_rate):
        # Gradient descent update
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
    
    def train(self, X, Y, iterations=1000, learning_rate=0.01):
        # Training loop
        for i in range(iterations):
            # Forward propagation
            A = self.forward(X)
            
            # Compute cost
            cost = self.compute_cost(A, Y)
            
            # Backpropagation
            dW, db = self.backward(X, A, Y)
            
            # Update parameters
            self.update_parameters(dW, db, learning_rate)
            
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost}")
    
    def predict(self, X):
        # Prediction using the forward propagation
        A = self.forward(X)
        return np.argmax(A, axis=1)

# Convert labels to one-hot encoded vectors
def one_hot_encode(y, num_classes):
    encoder = OneHotEncoder(sparse=False, categories='auto')
    y_reshaped = y.reshape(-1, 1)
    y_one_hot = encoder.fit_transform(y_reshaped)
    return y_one_hot

# Train and test the perceptron
def main():
    # Step 1: Prepare data
    X, y = prepare_data()
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # One-hot encode the labels (for multi-class classification)
    y_train_one_hot = one_hot_encode(y_train, 10)
    
    # Step 2: Initialize perceptron with 400 input units (20x20 images) and 10 output units (digits 0-9)
    perceptron = Perceptron(input_size=400, output_size=10)
    
    # Step 3: Train the perceptron using gradient descent
    perceptron.train(X_train, y_train_one_hot, iterations=2000, learning_rate=0.1)
    
    # Step 4: Test the perceptron
    predictions = perceptron.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
