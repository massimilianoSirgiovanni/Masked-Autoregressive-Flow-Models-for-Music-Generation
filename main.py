import pretty_midi
import torch

from manageMIDI import *
from sys import exit
from vaeModel import *
from train import *
from modelSelection import *
from mafModel import *
import h5py
import sqlite3
import pandas as pd

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Imposta le tue credenziali Spotify
client_id = 'f1e092078f9f4aac8ba0c57018645957'
client_secret = 'e0494501e9b84408803f0d9dcbfea3fb'

# Inizializza l'oggetto Spotipy
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))



def getGenre(song_name):
    # Cerca la canzone su Spotify
    result = sp.search(q=song_name, type='track', limit=1)

    # Estrai il genere dalla risposta
    track = result['tracks']['items'][0]

    artist = sp.artist(track["artists"][0]["external_urls"]["spotify"])
    return artist["genres"]

def getSongName(file_path):
    with h5py.File(file_path, 'r') as file:
        return file['metadata']['songs']['title'][0]



def loadSongGenres(directory, dictGenres):
    print(f"Loading Dataset from {directory} ...")
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(".h5")]:
            try:
                songName = getSongName(f"{dirpath}/{filename}")
                genres = getGenre(songName)
                for genre in genres:
                    if genre not in dictGenres:
                        dictGenres[genre] = 1
                    else:
                        dictGenres[genre] += 1
            except IndexError:
                pass

def countGenres(directory, starting_point="A", ending_point="Z"):
    if exists("./savedObjects/datasets/dictGenres"):
        dictGenres = loadVariableFromFile("./savedObjects/datasets/dictGenres")
    else:
        dictGenres = {}
    for i in range(ord(starting_point), ord(ending_point) + 1):
        loadSongGenres(f"{current_directory}/{directory}/{chr(i)}", dictGenres)
        saveVariableInFile("./savedObjects/datasets/dictGenres", dictGenres)
        print(f"Letter {chr(i)} downloaded")
    dictGenres = dict(sorted(dictGenres.items(), key=lambda item: item[1], reverse=True))
    saveVariableInFile("./savedObjects/datasets/dictGenres", dictGenres)
    return dictGenres

countGenres("./lmd_matched_h5")

'''with h5py.File(file_path, 'r') as file:
    # Ora puoi accedere ai dati nel file .h5
    # Ad esempio, puoi stampare le chiavi del gruppo radice
    elementi = list(file.keys())
    print("Chiavi nel gruppo radice:", elementi)

    for elemento in elementi:
        print("Dettagli di", elemento, ":")
        if isinstance(file[elemento], h5py.Group):
            print("Tipo: Gruppo")
            print(type(file[elemento]))
            tmp = list(file[elemento].keys())
            print(f"Chiavi nel gruppo {elemento}:", tmp)
            print(type(file[elemento][tmp[0]]))
        elif isinstance(file[elemento], h5py.Dataset):
            print("Tipo: Dataset")
            print("Forma del dataset:", file[elemento].shape)
            print("Tipo di dati del dataset:", file[elemento].dtype)
        print("-" * 30)'''
exit(0)

extendDataset = 0
torch.manual_seed(seeds[0])
torch.autograd.detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(f"Device: {device}")
print(f"program {choosedInstrument}")

print("--------------------Starting Execution---------------------")
print(f"{Fore.YELLOW}Seed = {seeds[0]}{Style.RESET_ALL}\n")

print(f"Choose Model:\n{Fore.LIGHTGREEN_EX}1. RNN;\n2. LSTM;\n3. MAF;{Style.RESET_ALL}\n")
value = '3'#input("Enter your choice here: ")
if value not in ["1", "2", "3"]:
    print(f"{Fore.RED}ERROR: The entered value does not match any selectable options{Style.RESET_ALL}")
    exit(-1)
if value == "1":
    choosedModel = VAE(nn.RNN)
    loss_function = loss_function_VAE
    choosedDirectory = directory_models_RNN
elif value == "2":
    choosedModel = VAE(nn.LSTM)
    loss_function = loss_function_VAE
    choosedDirectory = directory_models_LSTM
elif value == "3":
    choosedModel = MAF()
    loss_function = loss_function_maf
    choosedDirectory = directory_models_MAF
choosedModel.to(device)

tr_set=None;val_set=None;test_set = None;dataset = None
print(f"Enter the corresponding number to choose what you want to execute:\n{Fore.LIGHTGREEN_EX}1. Download and preprocessing on data\n2. Model Selection\n3. Train the choosen model\n4. Testing the choosen model\n5. Complete Execution{Style.RESET_ALL}")
value = '1'#input("Enter your choice here: ")

if value not in ["1", "2", "3", "4", "5"]:
    print(f"{Fore.RED}ERROR: The entered value does not match any selectable options{Style.RESET_ALL}")
    exit(-1)
elif value == "1":
    print(f"{Fore.LIGHTGREEN_EX}-------------PREPROCESSING------------------{Style.RESET_ALL}")
    if exists(f"./savedObjects/datasets/2_bar/dataset_complete_program={choosedInstrument}"):
        dataset = loadVariableFromFile(f"./savedObjects/datasets/2_bar/dataset_complete_program={choosedInstrument}")
    else:
        dataset = getSingleInstrumentDatabaseLMD(f"./savedObjects/datasets/2_bar", choosedInstrument)
    print(dataset.shape)
    tr_set, val_set, te_set = holdoutSplit(dataset, val_percentage=0.2, test_percentage=0.2)
    tr_set.to(device);val_set.to(device);te_set.to(device)
    print(f"Training Set Size: {tr_set.shape}")
    print(f"Validation Set Size: {val_set.shape}")
    forward = 'n'#input(f"{Fore.MAGENTA}Do you want to continue running the Model Selection? (y/n) {Style.RESET_ALL}")
    if forward.lower() == "y":
        value = "2"
    else:
        pass
        #exit(0)
elif value == "2":
    print(f"{Fore.LIGHTGREEN_EX}-------------MODEL SELECTION------------------{Style.RESET_ALL}")
    if tr_set == None or val_set  == None:
        if exists("./savedObjects/datasets/2_bar/dataset_complete_program=0"):
            dataset = loadVariableFromFile("./savedObjects/datasets/2_bar/dataset_complete_program=0")
        else:
            dataset = getSingleInstrumentDatabaseLMD(f"./savedObjects/datasets/2_bar", 0)
        tr_set, val_set, te_set = holdoutSplit(dataset, val_percentage=0.2, test_percentage=0.2)
        tr_set.to(device); val_set.to(device); te_set.to(device)
    forward = input(f"{Fore.MAGENTA}Do you want to continue running the Model Selection? (y/n) {Style.RESET_ALL}")
    modelSelection(tr_set, val_set, choosedModel, directory_models=choosedDirectory, patience=10)
    if forward.lower() == "y":
        value = "3"
    else:
        exit(0)
elif value == "3":
    print(f"{Fore.LIGHTGREEN_EX}-------------TRAINING------------------{Style.RESET_ALL}")
    forward = input(f"{Fore.MAGENTA}Do you want to continue running the Model Selection? (y/n) {Style.RESET_ALL}")
    if forward.lower() == "y":
        value = "4"
    else:
        exit(0)
elif value == "4":
    print(f"{Fore.LIGHTGREEN_EX}-------------TESTING------------------{Style.RESET_ALL}")

elif value == "5":
    exit(0)

if exists("./savedObjects/models/MAF_model"):
    trainObj = loadVariableFromFile("./savedObjects/models/MAF_model")
    print(trainObj)
else:
    choosedModel.parametersInitialization((131, 32), hidden_sizes=[50, 100, 80], n_layers=3, device=device, madeType="univariate")
    print(choosedModel)
    trainObj = trainingModel(choosedModel)

#trainObj.trainModel(tr_set, val_set, te_set, batch_size=1000, loss_function=loss_function, num_epochs=200, patience=20, learning_rate=0.01, file_path="./savedObjects/models/MAF_model")

trainObj.plot_loss()
print(f"Threshold: {trainObj.bestModel.threshold}")
print("Generation")
generateAndSaveASong(trainObj, tr_set[30:31, :, :], file_path=f"./output/trSet")
print("\n--------------------------------------------------\n")
generateAndSaveASong(trainObj, tr_set[60:61, :, :], file_path=f"./output/trSet2")
print("\n--------------------------------------------------\n")
print("Test")
generateAndSaveASong(trainObj, te_set[20:21, :, :], file_path=f"./output/test")
print("\n--------------------------------------------------\n")
generateAndSaveASong(trainObj, te_set[99:100, :, :], file_path=f"./output/test2")
print("\n--------------------------------------------------\n")
generateAndSaveASong(trainObj, file_path=f"./output/firstOriginalSong")

