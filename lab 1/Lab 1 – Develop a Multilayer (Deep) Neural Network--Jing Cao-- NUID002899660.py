# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load the Heart Disease dataset (using the correct UCIMLRepo fetch)
from ucimlrepo import fetch_ucirepo

# fetch dataset
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

# Convert the target variable to binary classification
y = np.where(y > 0, 1, 0)  # Convert multi-class labels to binary: 0 (no disease), 1 (disease)

# Preprocess data - fill missing values if necessary
X.fillna(X.mean(), inplace=True)

# One-hot encode categorical features
X = pd.get_dummies(X, columns=['sex', 'cp', 'restecg', 'exang', 'slope', 'thal'], drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature data (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the neural network model based on the given architecture
model = Sequential([
    Dense(10, activation='relu', input_shape=(X_train.shape[1],)),  # Layer 1: 10 neurons, ReLU activation
    Dense(8, activation='relu'),  # Layer 2: 8 neurons, ReLU activation
    Dense(8, activation='relu'),  # Layer 3: 8 neurons, ReLU activation
    Dense(4, activation='relu'),  # Layer 4: 4 neurons, ReLU activation
    Dense(1, activation='sigmoid')  # Layer 5: 1 neuron, Sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Make predictions
predictions = model.predict(X_test)

# Print the evaluation results and predictions
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
print(f'Predictions: {predictions[:5]}')
print(f'True Labels: {y_test[:5]}')
