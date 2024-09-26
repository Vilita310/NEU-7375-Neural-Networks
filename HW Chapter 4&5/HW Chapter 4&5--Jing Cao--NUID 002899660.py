import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

# Step 1: Define Classes

# Class for Activation
class Activation:
    def __init__(self, activation_type='sigmoid'):
        self.activation_type = activation_type
    
    def get_activation(self):
        if self.activation_type == 'sigmoid':
            return tf.keras.activations.sigmoid
        elif self.activation_type == 'relu':
            return tf.keras.activations.relu
        elif self.activation_type == 'softmax':
            return tf.keras.activations.softmax
        # Add more activations as needed

# Class for Parameters (weights and biases)
class Parameters:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.biases = np.zeros((1, output_dim))

# Class for a Neuron (single unit)
class Neuron:
    def __init__(self, weights, bias, activation_type='sigmoid'):
        self.weights = weights
        self.bias = bias
        self.activation = Activation(activation_type).get_activation()
        
    def activate(self, x):
        z = np.dot(x, self.weights) + self.bias
        return self.activation(z)

# Class for a Layer (combination of neurons)
class Layer:
    def __init__(self, units, input_dim, activation_type='sigmoid'):
        self.units = units
        self.input_dim = input_dim
        self.activation = Activation(activation_type).get_activation()
        self.layer = Dense(units, input_shape=(input_dim,), activation=self.activation)

    def __call__(self, x):
        return self.layer(x)

# Class for the Model (Neural Network with One Hidden Layer)
class Model:
    def __init__(self, input_dim, hidden_units, output_units):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_units = output_units

        # Create layers
        self.hidden_layer = Layer(hidden_units, input_dim, activation_type='sigmoid')
        self.output_layer = Layer(output_units, hidden_units, activation_type='softmax')
        
        # Use an Input layer for a cleaner initialization
        self.input_layer = Input(shape=(input_dim,))

    def forward(self, x):
        # Forward pass through hidden layer and output layer
        hidden_output = self.hidden_layer(x)
        output = self.output_layer(hidden_output)
        return output

    @property
    def trainable_weights(self):
        return self.hidden_layer.layer.trainable_weights + self.output_layer.layer.trainable_weights

# Class for the Loss Function
class LossFunction:
    def __init__(self, loss_type='cross_entropy'):
        self.loss_type = loss_type
    
    def get_loss(self):
        if self.loss_type == 'cross_entropy':
            return tf.keras.losses.CategoricalCrossentropy()
        # Add other loss functions as needed

# Class for Forward Propagation
class ForwardProp:
    def __init__(self, model):
        self.model = model

    def propagate(self, x):
        return self.model.forward(x)

# Class for Back Propagation and Gradient Descent
class GradDescent:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.optimizer = Adam(learning_rate)

    def compute_gradients(self, x, y, loss_fn):
        with tf.GradientTape() as tape:
            predictions = self.model.forward(x)
            loss = loss_fn(y, predictions)
        gradients = tape.gradient(loss, self.model.trainable_weights)
        return gradients, loss

    def apply_gradients(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

# Class for Training
class Training:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.loss_fn = LossFunction().get_loss()
        self.grad_descent = GradDescent(model, learning_rate)

    def train_step(self, x, y):
        gradients, loss = self.grad_descent.compute_gradients(x, y, self.loss_fn)
        self.grad_descent.apply_gradients(gradients)
        return loss

    def fit(self, x_train, y_train, epochs=10):
        for epoch in range(epochs):
            loss = self.train_step(x_train, y_train)
            print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')

# Step 2: Example Training Code
def main():
    # Example placeholders for data (MNIST-like, 28x28 images flattened)
    input_size = 784  # Flattened size of 28x28 image
    output_size = 10  # Number of classes (digits 0-9)
    hidden_units = 64  # Number of neurons in hidden layer

    # Random example data for demonstration (1000 samples)
    x_train = np.random.randn(1000, input_size).astype(np.float32)
    y_train = tf.keras.utils.to_categorical(np.random.randint(0, output_size, (1000,)), output_size)

    # Initialize the Model
    model = Model(input_dim=input_size, hidden_units=hidden_units, output_units=output_size)

    # Initialize Training
    trainer = Training(model, learning_rate=0.01)

    # Train the Model
    trainer.fit(x_train, y_train, epochs=10)

if __name__ == "__main__":
    main()
