import colorama
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import norm
from colorama import Fore, Style
import genreAnalysis


def precision_with_flatten(recon_x, x):
    recon_x = torch.flatten(recon_x)
    x = torch.flatten(x)
    return precision_score(recon_x, x)

def recall_with_flatten(recon_x, x):
    recon_x = torch.flatten(recon_x)
    x = torch.flatten(x)
    return recall_score(recon_x, x)

def f1_score_with_flatten(recon_x, x):
    recon_x = torch.flatten(recon_x)
    x = torch.flatten(x)
    return f1_score(recon_x, x)

def two_norm(recon_x, x):
    dim = x.shape[0]
    return norm(x - recon_x, p=2)/dim


###################### Metrics for evaluating songs ########################

def holdStatePerSong(tensor):
    return torch.sum(tensor[:, :, -2])/tensor.shape[0]

def silentStatePerSong(tensor):
    return torch.sum(tensor[:, :, -1])/tensor.shape[0]

def playedNotesPerSong(tensor):
    return torch.sum(tensor[:, :, :-2])/tensor.shape[0]

def highestPitch(tensor):
    index = torch.where(tensor[:, :, :-3] == 1)
    zero = torch.zeros_like(tensor[:, :, :-3]).float()
    zero[index[0], index[1], index[2]] = index[2].float()
    max = torch.max(torch.max(zero, dim=2).values, dim=1).values
    return torch.mean(max)

def lowestPitch(tensor):
    index = torch.where(tensor[:, :, :-3] == 1)
    zero = torch.zeros_like(tensor[:, :, :-3]).float()
    zero[:, :, :] = 128
    zero[index[0], index[1], index[2]] = index[2].float()
    min = torch.min(torch.min(zero, dim=2).values, dim=1).values
    return torch.mean(min)

def averagePitch(tensor):
    index = torch.where(tensor[:, :, :-3] == 1)
    zero = torch.zeros_like(tensor[:, :, :-3]).float()
    zero[index[0], index[1], index[2]] = index[2].float()
    sum = torch.sum(zero, dim=(1, 2))
    element_unique, count = torch.unique(index[0], return_counts=True)
    avaragePitchPerSong = sum[element_unique]/count
    return torch.mean(avaragePitchPerSong)

def completeAnalisysOnSongsSets(tensor, stringGenre=""):
    print("\n" + "-"*8 + f"{Fore.MAGENTA}COMPLETE ANALYSIS ON A SET OF {stringGenre}TRACKS{Style.RESET_ALL}" + "-"*8)
    print(f"Number of tracks: {Fore.LIGHTGREEN_EX}{tensor.shape[0]}{Style.RESET_ALL}")
    print(f"Bars per track: {Fore.LIGHTGREEN_EX}{tensor.shape[1]/16}{Style.RESET_ALL} bars\n")
    print(f"Number of held notes: {Fore.LIGHTGREEN_EX}{holdStatePerSong(tensor):.4f}{Style.RESET_ALL}")
    print(f"Number of silent notes: {Fore.LIGHTGREEN_EX}{silentStatePerSong(tensor):.4f}{Style.RESET_ALL}")
    print(f"Number of played notes: {Fore.LIGHTGREEN_EX}{playedNotesPerSong(tensor):.4f}{Style.RESET_ALL}")
    print(f"Highest pitch of played notes: {Fore.LIGHTGREEN_EX}{highestPitch(tensor):.4f}{Style.RESET_ALL}")
    print(f"Lowest pitch of played notes: {Fore.LIGHTGREEN_EX}{lowestPitch(tensor):.4f}{Style.RESET_ALL}")
    print(f"Mean pitch of played notes: {Fore.LIGHTGREEN_EX}{averagePitch(tensor):.4f}{Style.RESET_ALL}")


def completeAnalisysOnSingleGenre(tensor, tensor_y, genre):
    genre = genreAnalysis.convertGenreToNumber(genre)
    mask_genre = tensor_y == genre
    tensor = tensor[mask_genre]
    completeAnalisysOnSongsSets(tensor, stringGenre=f"{genreAnalysis.convertGenreToString(genre)} ")
