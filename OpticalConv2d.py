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
        batch_size = input.size(dim=0)
        input, kernel = self.process_inputs(input, self.kernel)

        input = input.repeat(1,self.output_channels,1,1)
        input = torch.reshape(input,(batch_size,self.output_channels,self.input_channels,self.input_size, self.input_size))
        kernel = kernel.repeat(batch_size,1,1,1)
        kernel = torch.reshape(kernel,(batch_size,self.output_channels,self.input_channels,self.input_size, self.input_size))

        output = self.opt.optConv2d(input, kernel, pseudo_negativity=self.pseudo_negativity)
        output = torch.sum(output, dim=2,dtype=torch.float32)
        return output

