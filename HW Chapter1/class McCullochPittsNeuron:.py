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


# Interactive function to get user input and simulate the neuron
def simulate_interactive_neuron():
    print("Welcome to the McCulloch and Pitts Neuron Simulation!")

    # Step 1: Get the number of inputs from the user
    num_inputs = int(input("Enter the number of inputs to the neuron: "))

    # Step 2: Get the inputs from the user
    inputs = []
    for i in range(num_inputs):
        inp = int(input(f"Enter binary input {i+1} (0 or 1): "))
        while inp not in [0, 1]:  # Ensure the input is binary
            print("Input must be 0 or 1.")
            inp = int(input(f"Enter binary input {i+1} (0 or 1): "))
        inputs.append(inp)

    # Step 3: Ask the user if they want custom weights
    use_custom_weights = input("Do you want to provide custom weights? (y/n): ").lower() == 'y'
    
    # Step 4: Create the neuron and set the weights if needed
    neuron = McCullochPittsNeuron(num_inputs=num_inputs)

    if use_custom_weights:
        weights = []
        for i in range(num_inputs):
            weight = int(input(f"Enter weight for input {i+1}: "))
            weights.append(weight)
        neuron.set_weights(weights)

    # Step 5: Get the threshold from the user
    threshold = int(input("Enter the threshold for the neuron: "))
    neuron.threshold = threshold

    # Display the current configuration
    print(f"\nNeuron Configuration: {neuron}")

    # Step 6: Compute the output based on user inputs
    output = neuron.forward(inputs)
    print(f"\nFor inputs {inputs}, the neuron output is: {output}")


if __name__ == "__main__":
    simulate_interactive_neuron()
