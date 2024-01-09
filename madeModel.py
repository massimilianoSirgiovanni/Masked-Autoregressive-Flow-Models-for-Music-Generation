from typing import List
import torch
import torch.nn as nn
from maskedLinear import *
import numpy


class MADE(nn.Module):
    def __init__(self, input_size: tuple[int], num_genres: int = 0, seed=0):
        super(MADE, self).__init__()
        self.num_seq = input_size[1]
        self.num_features = input_size[0]
        self.m = {}
        self.seed = seed
        self.output_seq = self.num_seq
        self.num_genres = num_genres


    def forward(self, x, genres):
        for layer in self.layers:
            x = layer(x, genres)
        return x

    def generate(self, n_samples=1, u=None, genres=None):
        u = torch.randn((n_samples, self.num_features, self.num_seq)) if u is None else u
        x = torch.zeros((u.shape[0], self.num_features, self.num_seq))
        for idx_seq in range(0, self.num_seq):
            out = self.forward(x, genres)
            mu, logp = torch.chunk(out, 2, dim=2); del out
            x[:, :, idx_seq] = mu[:, :, idx_seq] + torch.exp(-0.5 * logp[:, :, idx_seq]) * u[:, :, idx_seq]; del mu, logp
        return x

    def createMasks(self):
        numLayer = len(self.dim_list) - 1  # It not consider the ReLu layers

        rng = numpy.random.RandomState(self.seed)
        self.seed = self.seed + 1

        in_size = self.dim_list[0]
        self.m[-1] = numpy.arange(in_size)
        for l in range(numLayer):
            self.m[l] = rng.randint(self.m[l - 1].min(), in_size-1, size=self.dim_list[l+1])
        masks = [self.m[l - 1][None, :] <= self.m[l][:, None] for l in range(numLayer)]
        masks.append(self.m[numLayer - 1][None, :] < self.m[-1][:, None])
        layers = [l for l in self.layers if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

class MADEUnivariate(MADE):
    def __init__(self, input_size: tuple[int], hidden_sizes: List[int], num_genres: int = 0, seed=0):
        super().__init__(input_size, num_genres=num_genres, seed=seed)
        self.hidden_sizes = hidden_sizes
        self.layers = nn.ModuleList()
        self.dim_list = [self.num_seq, *hidden_sizes, 2*self.output_seq]
        for i in range(len(self.dim_list) - 2):
            self.layers.append(MaskedLinearUnivariate(in_features=self.dim_list[i], out_features=self.dim_list[i + 1], num_genres=self.num_genres))
            self.layers.append(ReLU())
        self.layers.append(MaskedLinearUnivariate(in_features=self.dim_list[-2], out_features=self.dim_list[-1], num_genres=self.num_genres))
        self.createMasks()






class MADEDifferentMaskDifferentWeight(MADE):
    def __init__(self, input_size: tuple[int], hidden_sizes: List[int], num_genres: int = 0, seed=0):
        super().__init__(input_size, num_genres=num_genres, seed=seed)
        self.hidden_sizes = hidden_sizes
        self.layers = nn.ModuleList()
        self.dim_list = [self.num_seq, *hidden_sizes, 2*self.output_seq]
        for i in range(len(self.dim_list) - 2):
            self.layers.append(MaskedLinearMultinotes(self.dim_list[i], self.num_features,  self.dim_list[i + 1], num_genres=self.num_genres))
            self.layers.append(ReLU())
        self.layers.append(MaskedLinearMultinotes(self.dim_list[-2], self.num_features, self.dim_list[-1], num_genres=self.num_genres))
        self.createMasks()



class MADEMultivariate(MADE):
    def __init__(self, input_size: tuple[int], hidden_sizes: List[tuple[int]], num_genres: int = 0, seed=0):
        super(MADEMultivariate, self).__init__(input_size, num_genres=num_genres, seed=seed)
        self.layers = nn.ModuleList()
        self.dim_list = [self.num_seq]
        self.dim_list.extend([t[1] for t in hidden_sizes])
        self.dim_list.append(self.output_seq*2)
        self.dim_list_note = [self.num_features]
        self.dim_list_note.extend([t[0] for t in hidden_sizes])
        self.dim_list_note.append(self.num_features)
        print(hidden_sizes)
        print(self.dim_list)
        print(self.dim_list_note)
        self.layers = nn.ModuleList()
        for i in range(len(self.dim_list) - 2):
            self.layers.append(MaskedLinearMultivariate((self.dim_list_note[i], self.dim_list[i]), (self.dim_list_note[i+1], self.dim_list[i + 1]), num_genres=self.num_genres))
            self.layers.append(ReLU())
        self.layers.append(MaskedLinearMultivariate((self.dim_list_note[-2], self.dim_list[-2]), (self.dim_list_note[-1], self.dim_list[-1]), num_genres=self.num_genres))
        self.createMasks()

class ReLU(nn.ReLU):
    def forward(self, input, additional_arg=None):
        # To avoid modifying the forward in MADE, this class was created to replace nn.ReLU() and ignore the y
        return super(ReLU, self).forward(input)