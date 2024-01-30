from madeModel import MADEMultivariate, MADESharedWeights, MADEDifferentNotesDifferentWeights
from typing import List
from manageMIDI import binarize_predictions
from config import choosedDevice
from torch.nn import Module, ModuleList
from colorama import Fore, Style
from torch import sum, exp, eye, permute, zeros, mean, seed, manual_seed, randn, no_grad, Tensor
from numpy import pi, log



class MAFLayer(Module):
    def __init__(self,  input_size: tuple[int], hidden_sizes: List, num_genres: int = 0, madeType="shared", activationLayer='relu'):
        super(MAFLayer, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.madeType = madeType
        self.num_genres = num_genres
        if madeType == "multivariate":
            self.made = MADEMultivariate(input_size, hidden_sizes, num_genres=num_genres, activationLayer=activationLayer)
        elif madeType == "dndw":
            self.made = MADEDifferentNotesDifferentWeights(input_size, hidden_sizes, num_genres=num_genres, activationLayer=activationLayer)
        elif madeType == "shared":
            self.made = MADESharedWeights(input_size, hidden_sizes, num_genres=num_genres, activationLayer=activationLayer)
        else:
            self.madeType = "shared"
            print(f"{Fore.RED}WARNING! The string {Fore.LIGHTGREEN_EX}\"{madeType}\"{Fore.RED} does not match any MADE type, therefore was used the default type --> {Fore.LIGHTGREEN_EX}\"univariate\"{Style.RESET_ALL}")
            self.made = MADESharedWeights(input_size, hidden_sizes, num_genres=num_genres, activationLayer=activationLayer)

    def forward(self, x, genres):
        mu, log_p = self.made(x, genres)
        u = (x - mu) * exp(0.5 * log_p)
        return u, 0.5 * sum(log_p, dim=(1, 2))

    def generate(self, n_samples=1, u=None, genres=None):
        return self.made.generate(n_samples, u, genres=genres)


class MAF(Module):
    def __init__(self,):
        super().__init__()
        self.initialization = False
#
    def parametersInitialization(self,  input_size: tuple[int], hidden_sizes: List, n_layers: int, num_genres=0, madeType="shared", activationLayer='relu'):
        if self.initialization == False:
            self.initialization = True
            self.input_size = input_size
            self.n_layers = n_layers
            self.hidden_sizes = hidden_sizes
            self.madeType = madeType.lower()
            self.num_genres = num_genres
            self.layers = ModuleList()
            for _ in range(n_layers):
                self.layers.append(MAFLayer(self.input_size, self.hidden_sizes, num_genres=self.num_genres, madeType=self.madeType, activationLayer=activationLayer))
            self.madeType = self.layers[0].madeType
            self.threshold = 0.5
        else:
            print(f"{Fore.RED}WARNING! Parameters where already initialized{Style.RESET_ALL}")




    def oneHotEncoding(self, genres):
        if (self.num_genres > 0 and genres is not None) or len(genres.shape) < 2:
            genres = eye(self.num_genres)[genres.tolist()]
            if len(genres.shape) == 1:
                genres = genres.unsqueeze(0)
        return genres

    def forward(self, x, genres=None):
        u = permute(x, (0, 2, 1))
        genres = self.oneHotEncoding(genres).to(choosedDevice)
        log_det_sum = zeros(u.shape[0]).to(choosedDevice)
        for layer in self.layers:
            u, log_det = layer(u, genres)
            log_det_sum += log_det
        u = permute(u, (0, 2, 1))
        return u, self.negativeLogLikelihood(u, log_det_sum)

    def negativeLogLikelihood(self, u, log_det):
        negloglik_loss = (0.5 * sum(u ** 2, dim=(1, 2)))
        negloglik_loss += 0.5 * u.shape[1] * u.shape[2] * log(2 * pi)
        negloglik_loss -= log_det
        negloglik_loss = mean(negloglik_loss)
        return negloglik_loss


    def generate(self, n_samples=1, u=None, genres=Tensor([0]), choosedSeed=None):
        if choosedSeed == None:
            seed()
        else:
            manual_seed(choosedSeed)
        x = randn(n_samples, self.input_size[0], self.input_size[1]).to(choosedDevice) if u is None else permute(u, (0, 2, 1))
        if genres.shape[0] == 1:
            genres = self.oneHotEncoding(genres).to(choosedDevice)
        elif len(genres.shape) == 1:
            genres = genres.unsqueeze(0)
        self.eval()
        with no_grad():
            for layer in self.layers[::-1]:
                x = layer.generate(n_samples, x, genres=genres)
        x = permute(x, (0, 2, 1))
        x = binarize_predictions(x, self.threshold)
        return x

    def __str__(self):
        string = f"MAF model: \n >  madeType=\"{self.madeType}\"\n >  input_size={self.input_size}\n >  n_layers={self.n_layers}\n >  hidden_sizes={self.hidden_sizes}"
        return string