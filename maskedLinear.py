import torch
import torch.nn as nn
import math

class MaskedLinear(nn.Module):

    def forward(self, x):
        pass

    def set_mask(self, mask):
        self.mask = torch.tensor(mask)


class MaskedLinearUnivariate(nn.Linear, MaskedLinear):
    def __init__(self, in_features: int, out_features: int, device, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias)
        self.input_size = in_features
        self.output_size = out_features
        self.mask = None
        self.device = device

    def forward(self, x):
        output = torch.nn.functional.linear(x, torch.mul(self.weight, self.mask), self.bias)
        return output


class MaskedLinearMultinotes(MaskedLinear):
    def __init__(self, input_size: int, notes_size: int,  output_size: int, device, bias: bool = True, ) -> None:
        super().__init__()
        self.timestep_size = input_size
        self.notes_size = notes_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.Tensor(self.output_size, self.notes_size, self.timestep_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(self.output_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.mask = None

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight.reshape(self.output_size, -1))

            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        maskedW = torch.einsum("ont, ot -> ont", self.weight, self.mask)
        weighted_input = torch.einsum("bnt, ont -> bno", x, maskedW)
        return weighted_input + self.bias



class MaskedLinearMultivariate(MaskedLinear):
    def __init__(self, input_size: tuple, output_size: tuple, device, bias: bool = True, ) -> None:
        super().__init__()
        self.notes_size, self.timestep_size = input_size
        self.notes_out, self.timestep_out = output_size
        self.weight = nn.Parameter(torch.Tensor(self.notes_out, self.timestep_out, self.notes_size, self.timestep_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(self.notes_out * self.timestep_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.mask = None



    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight.reshape(self.weight.shape[0]*self.weight.shape[1], -1))

            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        maskedWeight = torch.mul(self.weight, self.mask)
        maskedWeight = maskedWeight.reshape(maskedWeight.shape[0]*maskedWeight.shape[1], -1)
        output = torch.nn.functional.linear(x.reshape(x.shape[0], -1), maskedWeight, self.bias)
        output = output.reshape(-1, self.weight.shape[0], self.weight.shape[1])
        return output