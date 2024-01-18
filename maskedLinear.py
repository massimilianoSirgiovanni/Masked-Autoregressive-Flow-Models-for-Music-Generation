import torch
import torch.nn as nn
import math

class MaskedLinear(nn.Module):

    def forward(self, x, y):
        pass

    def set_mask(self, mask):
        self.mask = torch.tensor(mask)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight.reshape(self.output_size, -1))

            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


class MaskedLinearUnivariate(MaskedLinear):
    def __init__(self, in_features: int, out_features: int, num_genres: int = 0, bias: bool = True) -> None:
        super().__init__()
        self.input_size = in_features
        self.output_size = out_features
        self.num_genres = num_genres
        self.weight = nn.Parameter(torch.Tensor(self.num_genres, self.output_size, self.input_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(self.output_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.mask = None

    def forward(self, x, y):
        maskedWeight = torch.einsum("got, ot -> got", self.weight, self.mask)
        genreWeight = torch.einsum("bg, got -> bot", y, maskedWeight)
        output = torch.einsum("bnt, bot -> bno", x.float(), genreWeight)
        if self.bias != None:
            return output + self.bias
        else:
            return output


class MaskedLinearMultinotes(MaskedLinear):
    def __init__(self, input_size: int, notes_size: int,  output_size: int, num_genres: int = 0, bias: bool = True, ) -> None:
        super().__init__()
        self.timestep_size = input_size
        self.notes_size = notes_size
        self.output_size = output_size
        self.num_genres = num_genres
        self.weight = nn.Parameter(torch.Tensor(self.num_genres, self.output_size, self.notes_size, self.timestep_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(self.output_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.mask = None



    def forward(self, x, y):
        maskedWeight = torch.einsum("gont, ot -> gont", self.weight, self.mask)
        genreWeight = torch.einsum("bg, gont -> bont", y, maskedWeight)
        return torch.einsum("bnt, bont -> bno", x.float(), genreWeight) + self.bias



class MaskedLinearMultivariate(MaskedLinear):
    def __init__(self, input_size: tuple, output_size: tuple, num_genres: int = 0, bias: bool = True, ) -> None:
        super().__init__()
        self.notes_size, self.timestep_size = input_size
        self.notes_out, self.timestep_out = output_size
        self.num_genres = num_genres
        self.weight = nn.Parameter(torch.Tensor(self.num_genres, self.notes_out, self.timestep_out, self.notes_size, self.timestep_size))
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

    def forward(self, x, y):
        maskedWeight = torch.einsum("gdont, ot -> gdont", self.weight, self.mask)
        maskedWeight = torch.einsum("gdont, bg -> bdont", maskedWeight, y)
        return torch.einsum("bnt, bdont -> bdo", x.float(), maskedWeight) + self.bias