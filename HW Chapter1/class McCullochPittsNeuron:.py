class McCullochPittsNeuron:
    def __init__(self, num_inputs, threshold=1):
        """
        Initialize a McCulloch and Pitts Neuron.
        
        :param num_inputs: The number of inputs to the neuron.
        :param threshold: The threshold for firing the neuron. Default is 1.
        """
        self.num_inputs = num_inputs
        self.weights = [1] * num_inputs  # Initial binary weights (1 for simplicity)
        self.threshold = threshold

    def set_weights(self, weights):
        """
        Set custom weights for the inputs.
        
        :param weights: A list of weights (same length as inputs).
        """
        if len(weights) != self.num_inputs:
            raise ValueError("Number of weights must match number of inputs.")
        self.weights = weights

    def activation_function(self, weighted_sum):
        """
        A simple step activation function.
        
        :param weighted_sum: The sum of the weighted inputs.
        :return: 1 if the sum exceeds the threshold, otherwise 0.
        """
        return 1 if weighted_sum >= self.threshold else 0

    def forward(self, inputs):
        """
        Perform a forward pass through the neuron to compute the output.
        
        :param inputs: A list of binary inputs (0 or 1).
        :return: The output of the neuron (0 or 1).
        """
        if len(inputs) != self.num_inputs:
            raise ValueError("Number of inputs must match the number of neuron inputs.")

        # Calculate the weighted sum of inputs
        weighted_sum = sum(w * inp for w, inp in zip(self.weights, inputs))
        
        # Apply the activation function (step function)
        return self.activation_function(weighted_sum)

    def __str__(self):
        return f"Neuron(weights={self.weights}, threshold={self.threshold})"


# Test the McCulloch and Pitts neuron
def simulate_neuron():
    # Example 1: Simple AND gate with 2 inputs
    print("Simulating a McCulloch and Pitts Neuron for AND gate")
    neuron = McCullochPittsNeuron(num_inputs=2, threshold=2)

    # Test cases for AND gate behavior
    test_cases = [
        ([0, 0], 0),  # Expected output: 0
        ([0, 1], 0),  # Expected output: 0
        ([1, 0], 0),  # Expected output: 0
        ([1, 1], 1)   # Expected output: 1
    ]

    for inputs, expected_output in test_cases:
        output = neuron.forward(inputs)
        print(f"Inputs: {inputs}, Output: {output}, Expected: {expected_output}")

    # Example 2: Custom neuron with custom weights
    print("\nSimulating a neuron with custom weights and threshold")
    custom_neuron = McCullochPittsNeuron(num_inputs=3, threshold=2)
    custom_neuron.set_weights([1, 1, 1])  # Setting custom weights

    # Test case for custom neuron
    custom_test_cases = [
        ([1, 0, 1], 1),  # Expected output: 1
        ([0, 0, 1], 0),  # Expected output: 0
        ([1, 1, 0], 1),  # Expected output: 1
        ([0, 0, 0], 0)   # Expected output: 0
    ]

    for inputs, expected_output in custom_test_cases:
        output = custom_neuron.forward(inputs)
        print(f"Inputs: {inputs}, Output: {output}, Expected: {expected_output}")


if __name__ == "__main__":
    simulate_neuron()
