import numpy as np

def depthwise_convolution(input_image, kernel):
    """
    Apply depthwise convolution on the input image with a given kernel.
    """
    # Assuming the image is of size (H, W, C)
    h, w, c = input_image.shape
    kernel_height, kernel_width, _ = kernel.shape
    output_height = h - kernel_height + 1
    output_width = w - kernel_width + 1
    
    # Initialize an output array
    output_image = np.zeros((output_height, output_width, c))

    for i in range(c):  # Loop through each channel
        for j in range(output_height):
            for k in range(output_width):
                output_image[j, k, i] = np.sum(input_image[j:j+kernel_height, k:k+kernel_width, i] * kernel[:, :, i])
    
    return output_image

def pointwise_convolution(input_image, depthwise_output, num_filters):
    """
    Apply pointwise convolution on the output of depthwise convolution.
    """
    h, w, c = depthwise_output.shape
    output_image = np.zeros((h, w, num_filters))
    
    for i in range(num_filters):  # Loop through each output filter
        for j in range(h):
            for k in range(w):
                output_image[j, k, i] = np.sum(depthwise_output[j, k, :] * np.ones(c))  # Using 1x1xC kernels
                
    return output_image

def convolution(input_image, kernel, is_depthwise=True, num_filters=1):
    if is_depthwise:
        depthwise_output = depthwise_convolution(input_image, kernel)
        return pointwise_convolution(input_image, depthwise_output, num_filters)
    else:
        # Standard convolution
        h, w, c = input_image.shape
        kernel_height, kernel_width, _ = kernel.shape
        output_height = h - kernel_height + 1
        output_width = w - kernel_width + 1
        output_image = np.zeros((output_height, output_width, num_filters))

        for i in range(num_filters):
            for j in range(output_height):
                for k in range(output_width):
                    output_image[j, k, i] = np.sum(input_image[j:j+kernel_height, k:k+kernel_width, :] * kernel)
        
        return output_image

# Example usage:
image = np.random.rand(12, 12, 3)  # Random 12x12x3 image
kernel = np.random.rand(5, 5, 3)  # Random 5x5 kernel for each channel
is_depthwise = True  # Set to False for standard convolution

output_image = convolution(image, kernel, is_depthwise, num_filters=256)
print(output_image.shape)  # Should print (8, 8, 256) for depthwise and pointwise
