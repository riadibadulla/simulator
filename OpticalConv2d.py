from torch import nn
import torch
from Optics_simulation import Optics_simulation
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class OpticalConv2dNew(nn.Module):

    def __init__(self,input_channels,output_channels,kernel_size,pseudo_negativity=False,input_size=28):
        super().__init__()
        self.pseudo_negativity = pseudo_negativity
        self.input_channels, self.output_channels = input_channels, output_channels
        self.kernel_size = kernel_size
        kernel = torch.Tensor(output_channels,input_channels,kernel_size,kernel_size)
        self.kernel = nn.Parameter(kernel)
        nn.init.kaiming_uniform_(self.kernel)
        self.input_size =input_size
        self.opt = Optics_simulation(input_size)

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
        if small.shape != large.shape:
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

        if img.shape==kernel.shape:
            return img, kernel
        size_of_image = img.shape[2]
        size_of_kernel = kernel.shape[2]
        padding_size = abs(size_of_image - size_of_kernel) // 2
        if size_of_image > size_of_kernel:
            img, kernel = self.__pad(img,kernel,padding_size)
        else:
            kernel, img = self.__pad(kernel,img,padding_size)
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
        input, kernel = self.process_inputs(input, self.kernel)

        input = input.repeat(1,self.output_channels,1,1)
        input = torch.reshape(input,(batch_size,self.output_channels,self.input_channels,self.input_size, self.input_size))
        kernel = kernel.repeat(batch_size,1,1,1)
        kernel = torch.reshape(kernel,(batch_size,self.output_channels,self.input_channels,self.input_size, self.input_size))

        output = self.opt.optConv2d(input, kernel, pseudo_negativity=self.pseudo_negativity)
        output = torch.sum(output, dim=2,dtype=torch.float32)
        return output
