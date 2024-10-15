import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Example dataset (MNIST-like)
def load_data():
    # For the sake of this example, we simulate random data
    X = np.random.rand(1000, 784)  # 1000 samples, each with 784 features (like MNIST)
    y = np.random.randint(0, 10, 1000)  # 10 classes (digits 0-9)
    return X, y

# Normalization class for input normalization
class Normalization:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X):
        self.scaler.fit(X)

    def transform(self, X):
        return self.scaler.transform(X)

    def fit_transform(self, X):
        return self.scaler.fit_transform(X)

# Main function to create normalized training, validation, and testing sets
def main():
    # Load dataset
    X, y = load_data()

    # Split data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Initialize normalization
    normalization = Normalization()

    # Normalize the training set and apply the same transformation to validation and test sets
    X_train_norm = normalization.fit_transform(X_train)
    X_val_norm = normalization.transform(X_val)
    X_test_norm = normalization.transform(X_test)

    # Output dataset sizes
    print(f"Training set size: {X_train_norm.shape[0]}")
    print(f"Validation set size: {X_val_norm.shape[0]}")
    print(f"Test set size: {X_test_norm.shape[0]}")

if __name__ == "__main__":
    main()
