
# Optical Neural Networks

This library is a custom convolutional layer in PyTorch, which uses the simulation of the light 
propagation in the 4F device. 

If using PyTorch, the layer can be used in the similar way as the standrd Conv2D. It has to be imported as: 
**from optnn import OpticalConv2d**

OpticalConv2d(input_channels, output_channels, kernel_size, is_bias=True, pseudo_negativity=False, input_size=28)

input_size of the layer has to be passed, the default value for input is 28x28. 



