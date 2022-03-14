import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
from torch import Tensor
from typing import Optional, List, Tuple, Union
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from Optics_simulation import Optics_simulation
from joblib import Parallel, delayed
import math
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class OpticalConv2dNew(nn.Module):

    def __init__(self,input_channels,output_channels,kernel_size,input_size):
        super().__init__()
        self.opt = Optics_simulation(28)
        self.input_channels, self.output_channels = input_channels, output_channels
        self.kernel_size = kernel_size
        kernel = torch.Tensor(output_channels,input_channels,kernel_size,kernel_size)
        self.kernel = nn.Parameter(kernel)
        nn.init.kaiming_uniform_(self.kernel, a=math.sqrt(5))

    def forward(self,input):
        output = torch.zeros(size=(input.size(dim=0), self.kernel.size(dim=0), input.size(dim=2), input.size(dim=3))).to(device)
        for batch in range(input.size(dim=0)):
            for output_channel in range(output.shape[1]):
                for image in input[batch, :, :, :]:
                    input_channel = 0
                    output[batch, output_channel, :, :] = output[batch, output_channel, :, :] + self.opt.optConv2d(
                        image, self.kernel[output_channel, input_channel, :, :])
                    input_channel += 1
        return output

