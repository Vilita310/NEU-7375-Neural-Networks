import tensorflow as tf
from tensorflow.keras import Input, regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

# Define the Neural Network with L2 Regularization and Dropout
class NeuralNetworkWithRegularization:
    def __init__(self, input_size, hidden_units, output_units, l2_lambda=0.01, dropout_rate=0.5):
        # Input layer
        self.input_layer = Input(shape=(input_size,))
        
        # Hidden layer with L2 regularization
        self.hidden_layer = Dense(hidden_units, activation='relu', 
                                  kernel_regularizer=regularizers.l2(l2_lambda))(self.input_layer)
        
        # Dropout layer
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)(self.hidden_layer)
        
        # Output layer
        self.output_layer = Dense(output_units, activation='softmax')(self.dropout_layer)

        # Define the model
        self.model = tf.keras.Model(inputs=self.input_layer, outputs=self.output_layer)

        # Compile the model with a loss function and optimizer
        self.model.compile(optimizer=Adam(learning_rate=0.01), 
                           loss='categorical_crossentropy', 
                           metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val, epochs=10):
        # Train the model
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, verbose=2)

    def evaluate(self, X_test, y_test):
        # Evaluate the model
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")

# Sample Training, Validation, and Testing Code
def prepare_data():
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize the data
    X_train = X_train.reshape(-1, 28*28).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28*28).astype('float32') / 255.0

    # Split training set into training and validation (80% training, 20% validation)
    X_train, X_val = X_train[:48000], X_train[48000:]
    y_train, y_val = y_train[:48000], y_train[48000:]
    
    # One-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()

    # Initialize the model
    model = NeuralNetworkWithRegularization(input_size=784, hidden_units=64, output_units=10, l2_lambda=0.01, dropout_rate=0.5)

    # Train the model
    model.train(X_train, y_train, X_val, y_val, epochs=10)

    # Evaluate the model
    model.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()
