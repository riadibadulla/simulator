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
        self.opt = Optics_simulation(input_size)

    def __pad(self,large,small,padding_size):
        small = torch.nn.functional.pad(small, (padding_size,padding_size,padding_size,padding_size))
        if small.shape != large.shape:
            small = torch.nn.functional.pad(small, (0,1,0,1))
        return large,small

    def process_inputs(self,img, kernel):
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
        input, kernel = self.process_inputs(input, self.kernel)
        output = torch.zeros(size=(input.size(dim=0), self.kernel.size(dim=0), input.size(dim=2), input.size(dim=3))).to(device)
        for batch in range(input.size(dim=0)):
            for output_channel in range(output.shape[1]):
                for image in input[batch, :, :, :]:
                    input_channel = 0
                    output[batch, output_channel, :, :] = output[batch, output_channel, :, :] + self.opt.optConv2d(
                        image, kernel[output_channel, input_channel, :, :], pseudo_negativity=self.pseudo_negativity)
                    input_channel += 1
        return output

