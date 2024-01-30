from torch.nn import Module, Parameter, init
from torch import tensor, Tensor, randn, einsum, matmul
from math import sqrt
from config import choosedDevice

class MaskedLinear(Module):

    def forward(self, x, y):
        pass

    def set_mask(self, mask):
        self.mask = tensor(mask).to(choosedDevice)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight.reshape(self.output_size, -1))
            bound = 1 / sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


class MaskedLinearSharedWeights(MaskedLinear):
    # Shared weights between notes
    def __init__(self, in_features: int, out_features: int, num_genres: int = 0, bias: bool = True, ) -> None:
        super().__init__()
        self.input_size = in_features
        self.output_size = out_features
        self.num_genres = num_genres
        self.weight = Parameter(Tensor(self.num_genres, self.output_size, self.input_size).to(choosedDevice))
        if bias:
            self.bias = Parameter(randn(self.output_size).to(choosedDevice))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.mask = None

    def forward(self, x, y):
        maskedWeight = self.weight * self.mask.unsqueeze(0)
        maskedWeight = einsum("bg, got -> bot", y, maskedWeight)
        if self.bias != None:
            return einsum("bnt, bot -> bno", x.float(), maskedWeight) + self.bias
        else:
            return einsum("bnt, bot -> bno", x.float(), maskedWeight)


class MaskedLinearDifferentNotesDifferentWeights(MaskedLinear):
    def __init__(self, input_size: int, notes_size: int,  output_size: int, num_genres: int = 0, bias: bool = True, ) -> None:
        super().__init__()
        self.timestep_size = input_size
        self.notes_size = notes_size
        self.output_size = output_size
        self.num_genres = num_genres
        self.weight = Parameter(Tensor(self.num_genres, self.output_size, self.notes_size, self.timestep_size).to(choosedDevice))
        if bias:
            self.bias = Parameter(randn(self.output_size).to(choosedDevice))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.mask = None



    def forward(self, x, y):
        maskedWeight = self.weight * self.mask.unsqueeze(0).unsqueeze(2)
        maskedWeight = einsum("bg, gont -> bont", y, maskedWeight)

        if self.bias != None:
            return einsum("bnt, bont -> bno", x.float(), maskedWeight) + self.bias
        else:
            return einsum("bnt, bont -> bno", x.float(), maskedWeight)




class MaskedLinearMultivariate(MaskedLinear):
    def __init__(self, input_size: tuple, output_size: tuple, num_genres: int = 0, bias: bool = True, ) -> None:
        super().__init__()
        self.notes_size, self.timestep_size = input_size
        self.notes_out, self.timestep_out = output_size
        self.num_genres = num_genres
        self.output_size = self.notes_out * self.timestep_out
        self.weight = Parameter(Tensor(self.num_genres, self.notes_out, self.timestep_out, self.notes_size, self.timestep_size).to(choosedDevice))
        if bias:
            self.bias = Parameter(randn(self.notes_out, self.timestep_out).to(choosedDevice))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.mask = None

    def forward(self, x, y):
        maskedWeight = self.weight * self.mask.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        maskedWeight = einsum("gdont, bg -> bdont", maskedWeight, y)
        if self.bias != None:
            return einsum("bnt, bdont -> bdo", x.float(), maskedWeight) + self.bias
        else:
            return einsum("bnt, bdont -> bdo", x.float(), maskedWeight)