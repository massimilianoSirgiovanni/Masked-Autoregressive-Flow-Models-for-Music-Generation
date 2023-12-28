import numpy
import torch
from colorama import *
from madeModel import *
from typing import List
from manageMIDI import binarize_predictions
import torch.utils.data as data
from tqdm import tqdm



    
class MAFLayer(nn.Module):
    def __init__(self,  input_size: tuple[int], hidden_sizes: List, device, madeType="univariate"):
        super(MAFLayer, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.device = device
        self.madeType = madeType
        if madeType == "multivariate":
            self.made = MADEMultivariate(input_size, hidden_sizes, device=self.device)
        elif madeType == "multinotes":
            self.made = MADEDifferentMaskDifferentWeight(input_size, hidden_sizes , device=self.device)
        elif madeType == "univariate":
            self.made = MADEUnivariate(input_size, hidden_sizes, device=self.device)
        else:
            self.madeType = "univariate"
            print(f"{Fore.YELLOW}The string:{madeType} does not match any MADE type, therefore was used the default type :\"univariate\"{Style.RESET_ALL}")
            self.made = MADEUnivariate(input_size, hidden_sizes, device=self.device)

    def forward(self, x):
        out = self.made(x)
        mu, log_p = torch.chunk(out, 2, dim=2)
        mu.to(self.device); log_p.to(self.device)
        u = (x - mu) * torch.exp(0.5 * log_p)
        return u, 0.5*torch.sum(log_p, dim=(1, 2))

    def generate(self, n_samples=1, u=None):
        return self.made.generate(n_samples, u)


class MAF(nn.Module):
    def __init__(self,):
        super().__init__()
        self.initialization = False

    def parametersInitialization(self,  input_size: tuple[int], hidden_sizes: List, n_layers: int, num_genre=0, embedding_dim=10, device=None, madeType="univariate"):
        if self.initialization == False:
            self.initialization = True
            self.input_size = input_size
            self.n_layers = n_layers
            self.hidden_sizes = hidden_sizes
            self.madeType = madeType.lower()
            self.embedding_dim = embedding_dim
            self.num_genre = num_genre
            if self.num_genre > 0:
                self.embedded_input_size = (self.input_size[0], self.input_size[1] + self.embedding_dim)
                self.embedding = nn.Embedding(self.num_genre, self.embedding_dim)
            else:
                self.embedded_input_size = self.input_size
            if device == None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = device
            self.layers = nn.ModuleList()
            for _ in range(n_layers):
                self.layers.append(MAFLayer(self.embedded_input_size, self.hidden_sizes, device=self.device))
            self.madeType = self.layers[0].madeType
            self.threshold = 0.5
            self.output_layer = nn.Linear(self.embedded_input_size[1], self.input_size[1])
        else:
            print(f"{Fore.LIGHTMAGENTA_EX}Parameters where already initialized{Style.RESET_ALL}")



    def mergeInputGenre(self, x, genres):
        y = self.embedding(genres)
        y = y.unsqueeze(1).expand(-1, x.shape[1], -1)
        u = torch.cat((x, y), dim=-1)
        return u

    def forward(self, x, genres=None):
        x = torch.permute(x, (0, 2, 1))
        if self.num_genre > 0 and genres is not None:
            mergedX = self.mergeInputGenre(x, genres)
            u = mergedX
        else:
            u = x
        log_det_sum = torch.zeros(u.shape[0])
        for layer in self.layers:
            u, log_det = layer(u)
            log_det_sum += log_det
        if self.num_genre > 0 and genres is not None:
            output = self.output_layer(u)
        output = torch.permute(output, (0, 2, 1))
        return output, self.negativeLogLikelihood(output, x, log_det_sum)

    def negativeLogLikelihood(self, u, x, log_det):
        negloglik_loss = - 0.5 * u.shape[1] * u.shape[2] * numpy.log(2 * numpy.pi) - 0.5 * torch.sum(u ** 2, dim=(1, 2)) + log_det
        negloglik_loss = - torch.mean(negloglik_loss)
        return negloglik_loss# - torch.sum(torch.log(1-x)) - torch.sum(torch.log(x))


    def generate(self, n_samples=1, u=None, genres=None):
        torch.set_printoptions(profile="full")
        x = torch.randn(n_samples, self.input_size[0], self.input_size[1]) if u is None else torch.permute(u, (0, 2, 1))
        if self.num_genre > 0 and genres is not None:
            x = self.mergeInputGenre(x, genres)
        self.eval()
        with torch.no_grad():
            for layer in self.layers[::-1]:
                x = layer.generate(n_samples, x)
        if self.num_genre > 0 and genres is not None:
            x = self.output_layer(x)
        print(torch.permute(x, (0, 2, 1))[0:1, :, :])
        print(self.threshold)
        x = binarize_predictions(x, self.threshold)
        torch.set_printoptions(profile="default")
        return torch.permute(x, (0, 2, 1))

    def __str__(self):
        string = f"MAF model: \n >  madeType=\"{self.madeType}\"\n >  input_size={self.input_size}\n >  n_layers={self.n_layers}\n >  hidden_sizes={self.hidden_sizes}"
        return string



def loss_function_maf(output, x, beta=0.1):
    z, nll = output

    return nll