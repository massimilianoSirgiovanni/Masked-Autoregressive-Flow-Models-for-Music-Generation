import colorama
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import norm
from colorama import Fore, Style
import genreAnalysis


def precision_with_flatten(recon_x, x):
    recon_x = torch.flatten(recon_x)
    x = torch.flatten(x)
    return precision_score(x, recon_x)

def recall_with_flatten(recon_x, x):
    recon_x = torch.flatten(recon_x)
    x = torch.flatten(x)
    return recall_score(x, recon_x)

def f1_score_with_flatten(recon_x, x):
    recon_x = torch.flatten(recon_x)
    x = torch.flatten(x)
    return f1_score(x, recon_x)

def two_norm(recon_x, x):
    dim = x.shape[0]
    return norm(x - recon_x, p=2)/dim


###################### Metrics for evaluating songs ########################

def holdStatePerSong(piano_roll):
    return torch.sum(piano_roll[:, :, -2])/piano_roll.shape[0]

def silentStatePerSong(piano_roll):
    return torch.sum(piano_roll[:, :, -1])/piano_roll.shape[0]

def playedNotesPerSong(piano_roll):
    return torch.sum(piano_roll[:, :, :-2])/piano_roll.shape[0]

def highestPitch(piano_roll):
    index = torch.where(piano_roll[:, :, :-3] == 1)
    zero = torch.zeros_like(piano_roll[:, :, :-3]).float()
    zero[index[0], index[1], index[2]] = index[2].float()
    max = torch.max(torch.max(zero, dim=2).values, dim=1).values
    return torch.mean(max)

def lowestPitch(piano_roll):
    index = torch.where(piano_roll[:, :, :-3] == 1)
    zero = torch.zeros_like(piano_roll[:, :, :-3]).float()
    zero[:, :, :] = 128
    zero[index[0], index[1], index[2]] = index[2].float()
    min = torch.min(torch.min(zero, dim=2).values, dim=1).values
    return torch.mean(min)

def averagePitch(piano_roll):
    index = torch.where(piano_roll[:, :, :-3] == 1)
    zero = torch.zeros_like(piano_roll[:, :, :-3]).float()
    zero[index[0], index[1], index[2]] = index[2].float()
    sum = torch.sum(zero, dim=(1, 2))
    element_unique, count = torch.unique(index[0], return_counts=True)
    avaragePitchPerSong = sum[element_unique]/count
    return torch.mean(avaragePitchPerSong)

def completeAnalisysOnSongsSets(piano_roll, stringGenre=""):
    print("\n" + "-"*8 + f"{Fore.MAGENTA}COMPLETE ANALYSIS ON A SET OF {Fore.LIGHTGREEN_EX}{stringGenre} {Fore.MAGENTA}TRACKS{Style.RESET_ALL}" + "-"*8)
    print(f"Number of tracks: {Fore.LIGHTGREEN_EX}{piano_roll.shape[0]}{Style.RESET_ALL}")
    print(f"Bars per track: {Fore.LIGHTGREEN_EX}{piano_roll.shape[1]/16}{Style.RESET_ALL} bars\n")
    print(f"Number of held notes: {Fore.LIGHTGREEN_EX}{holdStatePerSong(piano_roll):.4f}{Style.RESET_ALL}")
    print(f"Number of silent notes: {Fore.LIGHTGREEN_EX}{silentStatePerSong(piano_roll):.4f}{Style.RESET_ALL}")
    print(f"Number of played notes: {Fore.LIGHTGREEN_EX}{playedNotesPerSong(piano_roll):.4f}{Style.RESET_ALL}")
    print(f"Highest pitch of played notes: {Fore.LIGHTGREEN_EX}{highestPitch(piano_roll):.4f}{Style.RESET_ALL}")
    print(f"Lowest pitch of played notes: {Fore.LIGHTGREEN_EX}{lowestPitch(piano_roll):.4f}{Style.RESET_ALL}")
    print(f"Mean pitch of played notes: {Fore.LIGHTGREEN_EX}{averagePitch(piano_roll):.4f}{Style.RESET_ALL}")


def completeAnalisysOnSingleGenre(piano_roll, piano_roll_y, genre):
    genre = genreAnalysis.convertGenreToNumber(genre)
    mask_genre = piano_roll_y == genre
    piano_roll = piano_roll[mask_genre]
    completeAnalisysOnSongsSets(piano_roll, stringGenre=f"{genreAnalysis.convertGenreToString(genre)} ")

def notes_frequency(piano_roll):
    frequency = torch.sum(piano_roll, dim=(0, 1))/(piano_roll.shape[0]*piano_roll.shape[1])
    return frequency
