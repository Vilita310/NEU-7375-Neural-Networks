### Programming Assignment

import numpy as np

def convolution_2d(image, filter):
    # Get the dimensions of the input image and filter
    image_height, image_width = image.shape
    filter_height, filter_width = filter.shape
    
    # Calculate the dimensions of the output feature map
    output_height = image_height - filter_height + 1
    output_width = image_width - filter_width + 1
    
    # Initialize the output feature map with zeros
    output = np.zeros((output_height, output_width))
    
    # Perform the convolution operation
    for i in range(output_height):
        for j in range(output_width):
            # Extract the current region of interest from the input image
            region = image[i:i + filter_height, j:j + filter_width]
            # Element-wise multiplication and summation
            output[i, j] = np.sum(region * filter)
    
    return output

# Input 6x6 image
image = np.array([
    [1, 2, 3, 0, 1, 2],
    [0, 1, 2, 3, 1, 0],
    [3, 1, 0, 2, 3, 1],
    [2, 1, 2, 3, 0, 1],
    [0, 1, 3, 1, 2, 3],
    [2, 0, 1, 3, 2, 1]
])

# 3x3 filter
filter = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# Perform the convolution
output = convolution_2d(image, filter)

# Display the output
print("Convoluted Image:")
print(output)
