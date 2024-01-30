from sklearn.metrics import precision_score, recall_score, f1_score
from colorama import Fore, Style
from genreAnalysis import convertGenreToNumber, convertGenreToString
from torch import flatten, norm, sum, where, max, min, mean, unique, int8


def precision_with_flatten(recon_x, x):
    recon_x = flatten(recon_x)
    x = flatten(x)
    return precision_score(x, recon_x)

def recall_with_flatten(recon_x, x):
    recon_x = flatten(recon_x)
    x = flatten(x)
    return recall_score(x, recon_x)

def f1_score_with_flatten(recon_x, x):
    recon_x = flatten(recon_x)
    x = flatten(x)
    return f1_score(x, recon_x)

def two_norm(recon_x, x):
    dim = x.shape[0]
    return norm(x - recon_x, p=2)/dim


###################### Metrics for evaluating songs ########################

def holdStatePerSong(piano_roll):
    return sum(piano_roll[:, :, -1])/piano_roll.shape[0]

def silentStatePerSong(piano_roll):
    return where(sum(piano_roll, dim=2)==0)[0].shape[0]/piano_roll.shape[0]

def playedNotesPerSong(piano_roll):
    return sum(piano_roll[:, :, :-1])/piano_roll.shape[0]

def highestPitch(piano_roll):
    piano_roll = piano_roll[:, :, :-1].float()
    index = where(piano_roll == 1)
    piano_roll[index[0], index[1], index[2]] = index[2].to(int8).float()
    return mean(max(max(piano_roll.float(), dim=2).values, dim=1).values)

def lowestPitch(piano_roll):
    piano_roll = piano_roll[:, :, :-1].float()
    index = where(piano_roll == 1)
    piano_roll[:, :, :] = 128
    piano_roll[index[0], index[1], index[2]] = index[2].float()
    return mean(min(min(piano_roll, dim=2).values, dim=1).values)

def averagePitch(piano_roll):
    piano_roll = piano_roll[:, :, :-1]
    index = where(piano_roll == 1)
    piano_roll[index[0], index[1], index[2]] = index[2].to(int8)
    piano_roll = sum(piano_roll, dim=(1, 2))
    element_unique, count = unique(index[0], return_counts=True)
    return mean(piano_roll[element_unique]/count)

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
    genre = convertGenreToNumber(genre)
    mask_genre = piano_roll_y == genre
    piano_roll = piano_roll[mask_genre]
    completeAnalisysOnSongsSets(piano_roll, stringGenre=f"{convertGenreToString(genre)} ")

def notes_frequency(piano_roll):
    frequency = sum(piano_roll, dim=(0, 1))/(piano_roll.shape[0]*piano_roll.shape[1])
    return frequency
