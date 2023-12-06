from typing import List
import torch
import torch.nn as nn
from maskedLinear import *
import numpy

class MADEUnivariate(nn.Module):
    def __init__(self, input_size: tuple[int], hidden_sizes: List[int], device, seed=0):
        super().__init__()
        self.num_seq = input_size[1]
        self.num_notes = input_size[0]
        self.hidden_sizes = hidden_sizes
        self.layers = nn.ModuleList()
        self.dim_list = [self.num_seq, *hidden_sizes, 2*self.num_seq]
        self.device = device
        for i in range(len(self.dim_list) - 2):
            self.layers.append(MaskedLinearUnivariate(in_features=self.dim_list[i], out_features=self.dim_list[i + 1], device=device))
            self.layers.append(nn.ReLU())
        self.m = {}
        self.seed = seed
        self.layers.append(MaskedLinearUnivariate(in_features=self.dim_list[-2], out_features=self.dim_list[-1], device=device))
        self.createMasks()


    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return torch.sigmoid(output)

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
        layers = [l for l in self.layers if isinstance(l, MaskedLinearUnivariate)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def generate(self, n_samples=1, u=None):
        print("generation...")
        x = torch.zeros((n_samples, self.num_notes, self.num_seq))
        u = torch.randn((n_samples, self.num_notes, self.num_seq)) if u is None else u
        for i in range(0, self.num_seq):
            ord_seq = torch.Tensor(self.m[-1])
            idx_seq = torch.argwhere(ord_seq == i)[0, 0]
            out = self.forward(x)
            mu, logp = torch.chunk(out, 2, dim=2)
            x[:, :, idx_seq] = mu[:, :, idx_seq] + torch.exp(-0.5 * logp[:, :, idx_seq]) * u[:, :, idx_seq]
        return x

class MADEDifferentMaskDifferentWeight(nn.Module):
    def __init__(self, input_size: tuple[int], hidden_sizes: List[int], device):
        super().__init__()
        self.num_seq = input_size[1]
        self.num_notes = input_size[0]
        self.MADE = {}
        self.seed = 0
        for i in range(0, self.num_notes):
            self.MADE[i] = MADEUnivariate(input_size, hidden_sizes, device, self.seed)
            self.seed = (self.seed + 1)

    def forward(self, x):
        u = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*2))
        for i in range(0, self.num_notes):
            u[:, i:i+1, :] = self.MADE[i](x[:, i:i + 1, :])
        return u

    def generate(self, n_samples=1, u=None):
        x = torch.zeros((n_samples, self.num_notes, self.num_seq))
        u = torch.randn((n_samples, self.num_notes, self.num_seq)) if u is None else u
        for i in range(0, self.num_notes):
            print(self.MADE[i].generate(n_samples).shape)
            print(x[:, i, :].shape)
            x[:, i, :] = self.MADE[i].generate(n_samples)
        return x



class MADEMultivariate(nn.Module):
    def __init__(self, input_size: tuple[int], hidden_sizes: List[tuple[int]], device, seed=0):
        super(MADEMultivariate, self).__init__()
        self.num_seq = input_size[1]
        self.num_features = input_size[0]
        self.layers = nn.ModuleList()
        self.dim_list = [input_size, *hidden_sizes, (input_size[0], input_size[1]*2)]
        self.device = device
        self.layers = nn.ModuleList()
        for i in range(len(self.dim_list) - 2):
            self.layers.append(MaskedLinearMultivariate(self.dim_list[i], self.dim_list[i + 1], device=device))
            self.layers.append(nn.ReLU())
        self.seed = seed
        self.layers.append(MaskedLinearMultivariate(self.dim_list[-2], self.dim_list[-1], device=device))
        self.m_features = {}
        self.m_seq = {}
        self.createMask()

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return torch.sigmoid(output)

    def createMask(self):
        numLayer = len(self.dim_list) - 1  # It not consider the ReLu layers
        rng = numpy.random.RandomState(self.seed)
        self.seed = self.seed + 1

        in_size_feat = self.dim_list[0][0]
        self.m_features[-1] = rng.permutation(in_size_feat)
        in_size_seq = self.dim_list[0][1]
        self.m_seq[-1] = numpy.arange(in_size_seq)
        for l in range(numLayer):
            self.m_features[l] = rng.randint(self.m_features[l - 1].min(), in_size_feat - 1, size=self.dim_list[l + 1][0])
            self.m_seq[l] = rng.randint(self.m_seq[l - 1].min(), in_size_seq - 1, size=self.dim_list[l + 1][1])

        masks_features = [(self.m_features[l - 1][None, :] <= self.m_features[l][:, None])[:, None, :, None] for l in range(numLayer)]
        masks_features.append((self.m_features[numLayer - 1][None, :] < self.m_features[-1][:, None])[:, None, : ,None])

        masks_seq = [(self.m_seq[l - 1][None, :] <= self.m_seq[l][:, None])[None, :, None, :] for l in range(numLayer)]
        masks_seq.append((self.m_seq[numLayer - 1][None, :] < self.m_seq[-1][:, None])[None, :, None, :])

        masks = [masks_features[l] * masks_seq[l] for l in range(numLayer+1)]
        layers = [l for l in self.layers if isinstance(l, MaskedLinearMultivariate)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def generate(self, n_samples=1, u=None):
        x = torch.zeros((n_samples, self.num_features, self.num_seq))
        u = torch.randn((n_samples, self.num_features, self.num_seq)) if u is None else u
        for i in range(0, self.num_seq):
            ord_seq = torch.Tensor(self.m_seq[-1])
            idx_seq = torch.argwhere(ord_seq == i)[0, 0]
            for j in range(0, self.num_features):
                out = self.forward(x)
                mu, logp = torch.chunk(out, 2, dim=2)
                ord_feat = torch.Tensor(self.m_features[-1])
                idx_feat = torch.argwhere(ord_feat == j)[0, 0]
                x[:, idx_feat, idx_seq] = mu[:, idx_feat, idx_seq] + torch.exp(-0.5 * logp[:, idx_feat, idx_seq]) * u[:, idx_feat, idx_seq]
        return x