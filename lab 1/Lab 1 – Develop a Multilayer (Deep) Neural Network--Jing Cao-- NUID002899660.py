import zipfile
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Unzip the dataset
with zipfile.ZipFile('/path_to_your_file/heart+disease.zip', 'r') as zip_ref:
    zip_ref.extractall("/path_to_extract/heart_disease/")

# Load the dataset
data_path = "/path_to_extract/heart_disease/heart.csv"  # Update with actual CSV name inside the zip
df = pd.read_csv(data_path)

# Inspect data
print(df.head())

# Step 2: Preprocessing the Data
# Handle missing values if necessary (fill with median or mean)
df = df.fillna(df.median())

# Split the dataset into features and target
X = df.drop('target_column', axis=1)  # Replace 'target_column' with actual target column name
y = df['target_column']

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalize the data (StandardScaler)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Step 3: Designing the Neural Network
model = Sequential([
    Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification output layer
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Training the Model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

# Step 5: Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
