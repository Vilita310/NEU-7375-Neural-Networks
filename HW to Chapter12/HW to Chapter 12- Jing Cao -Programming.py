import numpy as np

class Softmax:
    def __init__(self):
        pass

    def forward(self, z):
        """
        Compute the softmax of vector z.
        
        Args:
            z (ndarray): Input array of logits
        
        Returns:
            ndarray: Softmax probabilities
        """
        exp_z = np.exp(z - np.max(z))  # subtracting max for numerical stability
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def backward(self, dz, softmax_output):
        """
        Compute the derivative of the softmax function.
        
        Args:
            dz (ndarray): Gradient of the loss with respect to the output
            softmax_output (ndarray): Output of the softmax function
            
        Returns:
            ndarray: Gradient of the loss with respect to the input
        """
        s = softmax_output.reshape(-1, 1)
        jacobian_matrix = np.diagflat(s) - np.dot(s, s.T)
        return np.dot(jacobian_matrix, dz)

# Example usage
if __name__ == "__main__":
    softmax = Softmax()
    z = np.array([2.0, 1.0, 0.1])
    
    # Forward pass (compute softmax probabilities)
    softmax_output = softmax.forward(z)
    print("Softmax output:", softmax_output)
    
    # Example of backward pass (with some hypothetical gradient dz)
    dz = np.array([0.1, 0.2, 0.3])
    grad_input = softmax.backward(dz, softmax_output)
    print("Gradient wrt input:", grad_input)
