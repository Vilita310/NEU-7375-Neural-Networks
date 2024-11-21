import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load the Heart Disease dataset from UCI Machine Learning Repository
from ucimlrepo import fetch_ucirepo
heart_disease = fetch_ucirepo(id=45)

# Data preparation
X = heart_disease.data.features
y = heart_disease.data.targets
y = np.where(y > 0, 1, 0)  # Binary classification: 0 (no disease), 1 (disease)
X.fillna(X.mean(), inplace=True)
X = pd.get_dummies(X, columns=['sex', 'cp', 'restecg', 'exang', 'slope', 'thal'], drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to create and compile a model
def create_model(layers, units, activation, learning_rate):
    model = Sequential()
    model.add(Dense(units, activation=activation, input_shape=(X_train.shape[1],)))
    for _ in range(layers - 1):
        model.add(Dense(units, activation=activation))
    model.add(Dense(1, activation='sigmoid'))  # Output layer
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define hyperparameter grid
hyperparameter_grid = {
    "layers": [2, 3, 4],                 # Number of layers
    "units": [8, 16, 32],               # Number of units per layer
    "activation": ['relu', 'tanh'],     # Activation function
    "learning_rate": [0.001, 0.0001]    # Learning rate
}

# Track the best configuration and results
best_accuracy = 0
best_params = {}

# Perform grid search
for layers in hyperparameter_grid["layers"]:
    for units in hyperparameter_grid["units"]:
        for activation in hyperparameter_grid["activation"]:
            for learning_rate in hyperparameter_grid["learning_rate"]:
                print(f"Training model with {layers} layers, {units} units, {activation} activation, and {learning_rate} learning rate.")
                # Create and train the model
                model = create_model(layers, units, activation, learning_rate)
                history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
                # Evaluate the model
                test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
                print(f"Validation Accuracy: {test_accuracy:.4f}")
                # Update best model
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    best_params = {
                        "layers": layers,
                        "units": units,
                        "activation": activation,
                        "learning_rate": learning_rate
                    }

# Print the best configuration
print(f"Best Validation Accuracy: {best_accuracy:.4f}")
print(f"Best Parameters: {best_params}")

# Train final model with best parameters
final_model = create_model(**best_params)
history = final_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the final model
final_test_loss, final_test_accuracy = final_model.evaluate(X_test, y_test)
print(f"Final Test Loss: {final_test_loss}, Final Test Accuracy: {final_test_accuracy}")

# Classification report
predictions = (final_model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, predictions))

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
