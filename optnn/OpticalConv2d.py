from torch import nn
import torch
from .Optics_simulation import Optics_simulation
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import math
class OpticalConv2d(nn.Module):
    """A custom Optical Layer class

    :param input_channels: Number of input channels of the layer
    :type input_channels: int
    :param output_channels: Number of output channels of the layer
    :type outout_channels: int
    :param kernel_size: size of the squared kernel
    :type kernel_size: int
    :param pseudo_negativity: Should the layer use pseudo negativity (decreases the computation time twice)
    :type pseudo_negativity: bool
    :param input_size: Layer accepts only square inputs, this is the size of the sqaured input
    :type input_size: int
    """

    def __init__(self,input_channels,output_channels,kernel_size,is_bias=True,pseudo_negativity=False,input_size=28):
        super().__init__()
        self.pseudo_negativity = pseudo_negativity
        self.input_channels, self.output_channels = input_channels, output_channels
        self.kernel_size = kernel_size
        kernel = torch.Tensor(output_channels,input_channels,kernel_size,kernel_size)
        self.kernel = nn.Parameter(kernel)
        nn.init.kaiming_uniform_(self.kernel, a=math.sqrt(5))
        self.is_bias = is_bias
        if is_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
            bound = 1 / math.sqrt(fan_in)
            bias = torch.Tensor(output_channels,1,1)
            self.bias = nn.Parameter(bias)
            nn.init.uniform_(self.bias, -bound, bound)
        self.input_size =input_size
        self.beam_size_px = kernel_size if kernel_size>input_size else input_size+4
        self.opt = Optics_simulation(self.beam_size_px)

    def __pad(self,large,small,padding_size):
        """Pads the tensors, so they can be the same size

        :param large: larger tensor, which doesn't have to be padded
        :type large: torch.Tensor
        :param small: small tensor which needs to be padded to the size of large
        :type small: torch.Tensor
        :param padding_size: Padding size for the small tensor
        :type padding_size: int
        :return: accepted tensors of the same size
        :rtype: torch.Tensor, torch.Tensor
        """
        small = torch.nn.functional.pad(small, (padding_size,padding_size,padding_size,padding_size))
        if small[0,0,0].shape != large[0,0,0].shape:
            small = torch.nn.functional.pad(small, (0,1,0,1))
        return large,small

    def process_inputs(self,img, kernel):
        """Takes to tensors of image and kernel and pads image or kernel depending on which one is larger

        :param img: Image tensor in (batch_size, input_channels, x, y) shape
        :type img: torch.Tensor
        :param kernel: kernel in (output_channels, input_channels, x ,y) shape
        :type kernel: torch.Tensor
        :return: image and kernel of the same size after padding one of them
        :rtype: torch.Tensor, torch.Tensor
        """
        img = torch.nn.functional.pad(img, (2, 2, 2, 2))
        if img.shape==kernel.shape:
            return img, kernel
        size_of_image = img.shape[2]
        size_of_kernel = kernel.shape[2]
        self.padding_size = abs(size_of_image - size_of_kernel) // 2
        if size_of_image > size_of_kernel:
            img, kernel = self.__pad(img,kernel,self.padding_size)
        else:
            kernel, img = self.__pad(kernel,img,self.padding_size)
        return img, kernel

    def forward(self,input):
        """Forward pass of the Optical convolution. It accepts the input tensor of shape (batch_size, input_channels, x, y)
        and takes the weights stored in this class in (output_channels, input_channels, x ,y) shape. Pad them to make
        both tensors same size. Then it turns both input and weight tensor to the same shape
        (batch_size, output_channels, input_channels, x,y) and performs the optical convolution using both tensors.
        Finally sums the output across the input channels to form the output of
        (batch_size, output_channels, x,y) shape.

        :param input: Image tensor in (batch_size, input_channels, x, y) shape
        :type input: torch.Tensor
        :return: output of (batch_size, output_channels, x, y) shape
        :rtype: torch.Tensor
        """
        batch_size = input.size(dim=0)
        #Padding either input or kernel
        input, kernel = self.process_inputs(input, self.kernel)
        input = input.repeat(1,self.output_channels,1,1)
        input = torch.reshape(input,(batch_size,self.output_channels,self.input_channels,self.beam_size_px,self.beam_size_px))
        kernel = kernel.repeat(batch_size,1,1,1)
        kernel = torch.reshape(kernel,(batch_size,self.output_channels,self.input_channels,self.beam_size_px,self.beam_size_px))
        output = self.opt.optConv2d(input, kernel, pseudo_negativity=self.pseudo_negativity)
        output = torch.sum(output, dim=2,dtype=torch.float32)

        #Upadding the input
        if self.kernel_size>self.input_size:
            output=output[:,:,self.padding_size:self.padding_size+self.input_size,self.padding_size:self.padding_size+self.input_size]
        #add bias
        if self.is_bias:
            output += self.bias.repeat(batch_size,1,1,1)
        return output
