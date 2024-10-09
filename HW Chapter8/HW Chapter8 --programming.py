import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

# Load MNIST dataset (or any other dataset you prefer)
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# Normalize the data (scaling pixel values between 0 and 1)
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# Reshape the data to make it compatible with fully connected layers
# From 28x28 images to 784 features (28*28)
X_train_full = X_train_full.reshape(X_train_full.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Step 2: Limit the dataset for prototype purposes
# For the prototype, we limit the number of samples to a manageable size (e.g., 1000 samples)
X_train_full = X_train_full[:1000]
y_train_full = y_train_full[:1000]

# Step 3: Split the data into training and validation sets
# We will keep 80% of the data for training and 20% for validation
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Step 4: Print the sizes of each set
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
