import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

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

# Build the neural network
model = Sequential([
    Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
predictions = model.predict(X_test)

# Output results
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
print(f'Predictions: {predictions[:5]}')
print(f'True Labels: {y_test[:5]}')
