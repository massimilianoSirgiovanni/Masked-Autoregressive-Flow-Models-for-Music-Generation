import math

import numpy
import torch
import torch.nn as nn
from typing import List

class MaskedLinear(nn.Module):
    def __init__(self, input_size: int, seq_len: int, output_size: int, device, bias: bool = True, ) -> None:
        #super().__init__(input_size*seq_len, output_size, bias)
        super(MaskedLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.relu = nn.ReLU()
        self.mask = self.createMask(seq_len)
        self.net = nn.LSTM(input_size=input_size*seq_len, hidden_size=output_size, num_layers=1,
                                    batch_first=True, bidirectional=False)
        self.hidden = nn.Linear(input_size, output_size)
        self.device = device

    def forward(self, x, hidden_0):
        """Apply masked linear transformation."""
        h_0 = self.hidden(hidden_0)
        #print(h_0.shape)
        vectMask = self.mask.unsqueeze(2).unsqueeze(0)
        maskedX = vectMask * x.unsqueeze(1)
        newX = maskedX.reshape(x.shape[0], x.shape[1], x.shape[1]*x.shape[2])
        #output = torch.nn.functional.linear(newX, self.weight, self.bias)
        output, hidden = self.net(newX, (h_0, h_0))
        return self.relu(output), h_0

    def createMask(self, seq_len):
        mask = torch.zeros(seq_len, seq_len)
        limit = 1
        for i in range(0, mask.shape[0]):
            mask[i, 0:limit] = 1
            limit += 1
        return mask


class MADE(nn.Module):
    def __init__(self, input_size: int, seq_len: int, hidden_sizes: List[int], device):
        super().__init__()
        self.layers = nn.ModuleList()
        dim_list = [input_size, *hidden_sizes, input_size*2]
        self.device = device
        self.initialHidden = nn.Parameter(torch.randn(1, input_size))
        for i in range(len(dim_list) - 2):
            self.layers.append(MaskedLinear(dim_list[i], seq_len, dim_list[i + 1], device=device))

        self.layers.append(MaskedLinear(dim_list[-2], seq_len, dim_list[-1], device=device))

    def forward(self, x):
        output = x
        hidden = self.initialHidden.repeat(1, x.shape[0], 1)
        #print(f"Intial Hidden: {hidden}")
        for layer in self.layers:
            output, hidden = layer(output, hidden)
        return torch.sigmoid(output)
    
class MAFLayer(nn.Module):
    def __init__(self,  input_size: int, seq_len: int, hidden_sizes: List[int], device):
        super(MAFLayer, self).__init__()
        self.device = device
        self.made = MADE(input_size, seq_len, hidden_sizes, device=self.device)

    def forward(self, x):
        out = self.made(x)
        mu, log_p = torch.chunk(out, 2, dim=2)
        #mu.to(self.device); log_p.to(self.device)
        u = (x - mu) * torch.exp(0.5 * log_p)  # residuo/deviazione standard

        return u, - torch.sum(log_p, dim=(1, 2))

class MAF(nn.Module):
    def __init__(self,):
        super().__init__()

    def parametersInitialization(self,  input_size: int, seq_len: int, n_layers: int, hidden_sizes: List[int], device=None):
        self.input_size = input_size
        self.n_layers = n_layers
        self.hidden_sizes = hidden_sizes
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(MAFLayer(input_size, seq_len, hidden_sizes, device=self.device))
        self.batch_norm = nn.BatchNorm1d(input_size)

    def forward(self, x):
        log_det_sum = torch.zeros(x.shape[0])
        for layer in self.layers:
            x, log_det = layer(x)
            x = torch.permute(x, (0, 2, 1))
            x = self.batch_norm(x)
            x = torch.permute(x, (0, 2, 1))
            log_det_sum += log_det
        return x, log_det_sum

    def __str__(self):
        string = f"MAF model: \n >  input_size={self.input_size}\n >  n_layers={self.n_layers}\n >  hidden_sizes={self.hidden_sizes}"
        return string



def loss_function_maf(output, x, beta=0.1):
    z, log_det = output
    negloglik_loss = 0.5 * (z ** 2).sum(dim=(1, 2))
    negloglik_loss += 0.5 * z.shape[1] * z.shape[2] * numpy.log(2 * math.pi)
    negloglik_loss -= log_det
    negloglik_loss = torch.mean(negloglik_loss)
    return negloglik_loss