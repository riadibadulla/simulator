import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
from torch import Tensor
from typing import Optional, List, Tuple, Union
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from Optics_simulation import Optics_simulation

opt = Optics_simulation(14)

class OpticalConv2d(nn.Conv2d):
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        # print(f"Input shape:{type(input.shape)}")
        # print(f"Weight shape: {weight.shape}")
        # print(F.conv2d(input, weight, bias, self.stride,
        #                 self.padding, self.dilation, self.groups).shape)

        # return F.conv2d(input, weight, bias, self.stride,
        #                 self.padding, self.dilation, self.groups)
        output = np.zeros(shape=(input.size(dim=0),weight.size(dim=0),input.size(dim=2),input.size(dim=3)))
        for batch in range(input.size(dim=0)):
            for output_channel in range(output.shape[1]):
                for image in input[batch,:,:,:]:
                    input_channel = 0
                    output[batch,output_channel,:,:]+= opt.optConv2d(image,weight[output_channel,input_channel,:,:])
                    input_channel+=1
        return torch.tensor(output)
