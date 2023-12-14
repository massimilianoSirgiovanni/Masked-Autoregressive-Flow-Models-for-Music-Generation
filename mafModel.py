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
            self.made = MADEDifferentMaskDifferentWeight(input_size, hidden_sizes, device=self.device)
        elif madeType == "univariate":
            self.made = MADEUnivariate(input_size, hidden_sizes, device=self.device)
        else:
            self.madeType = "univariate"
            print(f"{Fore.YELLOW}The string:{madeType} does not match any MADE type, therefore was used the default type :\"univariate\"{Style.RESET_ALL}")
            self.made = MADEUnivariate(input_size[1], hidden_sizes, device=self.device)

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

    def parametersInitialization(self,  input_size: tuple[int], hidden_sizes: List, n_layers: int, num_genres=0, embedding_dim=50, device=None, madeType="univariate"):
        if self.initialization == False:
            self.initialization = True
            self.input_size = input_size
            self.n_layers = n_layers
            self.hidden_sizes = hidden_sizes
            self.madeType = madeType.lower()
            self.num_genres = num_genres
            self.embedding_dim = embedding_dim
            if self.num_genres > 0:
                self.input_size[1] = self.input_size[1] + self.embedding_dim
                self.embedding = nn.Embedding(self.num_genres, self.embedding_dim)
            if device == None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = device
            self.layers = nn.ModuleList()
            for _ in range(n_layers):
                self.layers.append(MAFLayer(self.input_size, self.hidden_sizes, device=self.device, madeType=self.madeType))
            self.madeType = self.layers[0].madeType
            if self.madeType == "multivariate":
                self.batch_norm = nn.BatchNorm1d(input_size[1]*input_size[0])
            else:
                self.batch_norm = nn.BatchNorm1d(input_size[1])

            self.threshold = 0.7
            #self.threshold = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, requires_grad=True))
        else:
            print(f"{Fore.LIGHTMAGENTA_EX}Parameters where already initialized{Style.RESET_ALL}")



    def forward(self, x, genres=0):
        if self.num_genres > 0:
            y = self.embedding(genres)
            y = y.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat((x, y), dim=-1)
        log_det_sum = torch.zeros(x.shape[0])
        barch_size, seq_len, notes = x.shape
        for layer in self.layers:
            x = torch.permute(x, (0, 2, 1))
            x, log_det = layer(x)
            '''if self.madeType == "multivariate":
                x = self.batch_norm(x.reshape(barch_size, -1))
                x = x.reshape(-1, seq_len, notes)
            else:
                x = torch.permute(x, (0, 2, 1))
                x = self.batch_norm(x)'''
            x = torch.permute(x, (0, 2, 1))
            log_det_sum += log_det

        return x, self.negativeLogLikelihood(x, log_det_sum)

    def negativeLogLikelihood(self, z, log_det):
        negloglik_loss = - 0.5 * z.shape[1] * z.shape[2] * numpy.log(2 * numpy.pi) - 0.5 * torch.sum(z**2, dim=(1, 2)) + log_det
        negloglik_loss = - torch.mean(negloglik_loss)
        return negloglik_loss

    def chooseBestThreshold(self, tr_set, batch_size=500):
        #torch.set_printoptions(profile="full")
        dataset = data.DataLoader(tr_set, batch_size=batch_size, shuffle=True)
        self.eval()
        with torch.no_grad():
            globalThresholds = []
            for batch_data in tqdm(dataset, desc='Choosing Best Threshold: '):
                x_generated = self.genNoBin(u=batch_data)
                thresholds = numpy.arange(0.5, 1.0, 0.01)
                bestThresholds = []
                minValue = None
                for threshold in thresholds:
                    new_x = binarize_predictions(x_generated, threshold)
                    new_x = torch.permute(new_x, (0, 2, 1))
                    norm = torch.norm(batch_data - new_x, p=2)
                    if minValue == None or norm < minValue:
                        minValue = norm
                        bestThresholds = [threshold]
                    elif norm == minValue:
                        bestThresholds.append(threshold)
                print(bestThresholds)
                globalThresholds.append(numpy.median(bestThresholds))
        print(globalThresholds)
        self.threshold = numpy.mean(globalThresholds)
        print(bestThresholds)
        #torch.set_printoptions(profile="default")

    def generate(self, n_samples=1, u=None):
        torch.set_printoptions(profile="full")
        x = self.genNoBin(n_samples=n_samples, u=u)
        x = binarize_predictions(x, self.threshold)
        torch.set_printoptions(profile="default")
        return torch.permute(x, (0, 2, 1))

    def genNoBin(self, n_samples=1, u=None):

        x = torch.randn(n_samples, self.input_size[0], self.input_size[1]) if u is None else torch.permute(u, (0, 2, 1))
        self.eval()
        with torch.no_grad():
            for layer in self.layers[::-1]:
                x = layer.generate(n_samples, x)
        if u is not None:
            norm = torch.norm(torch.permute(u, (0, 2, 1)) - x, p=2)
            print(f"Similarity measure (two-norm) between input and midi generator: {norm}")

        return x

    def __str__(self):
        string = f"MAF model: \n >  madeType=\"{self.madeType}\"\n >  input_size={self.input_size}\n >  n_layers={self.n_layers}\n >  hidden_sizes={self.hidden_sizes}"
        return string



def loss_function_maf(output, x, beta=0.1):
    z, nll = output

    return nll