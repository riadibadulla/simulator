import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
from torch import Tensor
from typing import Optional, List, Tuple, Union
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from Optics_simulation import Optics_simulation
from joblib import Parallel, delayed
opt = Optics_simulation(28)
import math

class OpticalConv2d(nn.Conv2d):
    #TODO: Replace this class. Extend nn.Module like in https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77
    def _run_batch(self, batch,input,weight):
        for output_channel in range(self.output.shape[1]):
                for image in input[batch,:,:,:]:
                    input_channel = 0
                    self.output[batch,output_channel,:,:] = self.output[batch,output_channel,:,:] + opt.optConv2d(image,weight[output_channel,input_channel,:,:])
                    input_channel+=1

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        parallel = False
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        self.output = torch.zeros(size=(input.size(dim=0),weight.size(dim=0),input.size(dim=2),input.size(dim=3)))

        if parallel:
            Parallel(n_jobs=input.size(dim=0))(delayed(self._run_batch)(batch,input,weight) for batch in range(input.size(dim=0)))
        else:
            for batch in range(input.size(dim=0)):
                self._run_batch(batch,input, )
        return self.output


class OpticalConv2dNew(nn.Module):

    def _run_batch(self, batch,input,weight):
        for output_channel in range(self.output.shape[1]):
                for image in input[batch,:,:,:]:
                    input_channel = 0
                    self.output[batch,output_channel,:,:] = self.output[batch,output_channel,:,:] + opt.optConv2d(image,weight[output_channel,input_channel,:,:])
                    input_channel+=1

    def __init__(self,input_channels,output_channels,kernel_size):
        super().__init__()
        self.input_channels, self.output_channels = input_channels, output_channels
        self.kernel_size = kernel_size
        kernel = torch.Tensor(output_channels,input_channels,kernel_size,kernel_size)
        self.kernel = nn.Parameter(kernel)
        nn.init.kaiming_uniform_(self.kernel, a=math.sqrt(5))

    def forward(self,input):
        output = torch.empty(size=(input.size(dim=0), self.kernel.size(dim=0), input.size(dim=2), input.size(dim=3)))
        for batch in range(input.size(dim=0)):
            for output_channel in range(output.shape[1]):
                for image in input[batch, :, :, :]:
                    input_channel = 0
                    output[batch, output_channel, :, :] = output[batch, output_channel, :, :] + opt.optConv2d(
                        image, self.kernel[output_channel, input_channel, :, :])
                    input_channel += 1
        return output

