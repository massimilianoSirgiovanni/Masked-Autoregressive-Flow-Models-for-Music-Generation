import pretty_midi
import torch

from manageMIDI import *
from sys import exit
from vaeModel import *
from train import *
from modelSelection import *
from mafModel import *
from genreAnalysis import *
from collections import Counter
import pypianoroll
from config import *

import torch


#countGenres("./lmd_matched_h5")
'''loadLMD("./lpd/lpd_cleansed", ending_point='Z')
exit(0)'''
def estimateDatasetsSize():
    for instrument in range(0, 128):
        for i in range(ord("A"), ord("A") + 1):
            if os.path.isfile(f"./savedObjects/datasets/2_bar/{chr(i)}/dataset_program={instrument}"):
                input = loadVariableFromFile(f"./savedObjects/datasets/2_bar/{chr(i)}/dataset_program={instrument}")
                if input != None:
                    print(input[0].shape)
                else:
                    print(input)



#dataset, genres = loadVariableFromFile(f"./savedObjects/datasets/2_bar/dataset_complete_program=27")

dataset, genres = getSingleInstrumentDatabaseLMD(f"./savedObjects/datasets/2_bar", 0) # 25 27 29 30 33 35 48
print(dataset)
print(f"Piano Roll Dataset: {dataset.shape}")
print(f"Genre Labels: {genres.shape}")
exit()
extendDataset = 0
torch.manual_seed(seeds[0])
torch.autograd.detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"program {choosedInstrument}")
print(choosedGenres)
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

notes = range(0, 131)
print(notes)

tr_set=None;val_set=None;test_set = None;dataset = None
print(f"Enter the corresponding number to choose what you want to execute:\n{Fore.LIGHTGREEN_EX}1. Download and preprocessing on data\n2. Model Selection\n3. Train the choosen model\n4. Testing the choosen model\n5. Complete Execution{Style.RESET_ALL}")
value = '1'#input("Enter your choice here: ")

if value not in ["1", "2", "3", "4", "5"]:
    print(f"{Fore.RED}ERROR: The entered value does not match any selectable options{Style.RESET_ALL}")
    exit(-1)
elif value == "1":
    print(f"{Fore.LIGHTGREEN_EX}-------------PREPROCESSING------------------{Style.RESET_ALL}")
    if exists(f"./savedObjects/datasets/2_bar/dataset_complete_program={choosedInstrument}"):
        dataset, genres = loadVariableFromFile(f"./savedObjects/datasets/2_bar/dataset_complete_program={choosedInstrument}")
    else:
        dataset, genres = getSingleInstrumentDatabaseLMD(f"./savedObjects/datasets/2_bar", choosedInstrument)
    genres = gennreLabelToTensor(genres, choosedGenres)
    print(genres)
    tr_set, val_set, te_set = holdoutSplit(dataset[:, :, notes], genres, val_percentage=0.2, test_percentage=0.2)
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
            dataset, genres = loadVariableFromFile("./savedObjects/datasets/2_bar/dataset_complete_program=0")
        else:
            dataset, genres = getSingleInstrumentDatabaseLMD(f"./savedObjects/datasets/2_bar", choosedInstrument)
        genres = gennreLabelToTensor(genres, choosedGenres)
        tr_set, val_set, te_set = holdoutSplit(dataset, genres, val_percentage=0.2, test_percentage=0.2)
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
    #[(60, 50), (90, 80)]
    #[50, 80]
    choosedModel.parametersInitialization((len(notes), 32), hidden_sizes=[50, 80], embedding_dim=2, num_genre=len(choosedGenres), n_layers=3, device=device, madeType="univariate")
    print(choosedModel)
    trainObj = trainingModel(choosedModel)



#choosedModel(dataset, genres)
trainObj.trainModel(tr_set, val_set, batch_size=200, loss_function=loss_function, num_epochs=200, patience=20, learning_rate=0.01, file_path="./savedObjects/models/MAF_model")

trainObj.plot_loss()
trainObj.testModel(tr_set, set_name="Training Set", batch_size=1000)
trainObj.testModel(te_set, set_name="Test Set", batch_size=1000)
exit(0)


print("Generation")
print({label: idx for idx, label in enumerate(choosedGenres)})

generateAndSaveASong(trainObj.bestModel, tr_set.tensors[0][100:101, :, notes], genres=tr_set.tensors[1][100:101], file_path=f"./output/trSetOriginal", instrument=choosedInstrument)
exit(0)
print("\n--------------------------------------------------\n")
generateAndSaveASong(trainObj, tr_set.tensors[0][30:31, :, notes], genres=torch.tensor([1], dtype=torch.int8), file_path=f"./output/trSetCountry", instrument=choosedInstrument)
print("\n--------------------------------------------------\n")
exit(0)
generateAndSaveASong(trainObj, tr_set[60:61, :, notes], file_path=f"./output/trSet2", instrument=choosedInstrument)
print("\n--------------------------------------------------\n")
print("Test")
generateAndSaveASong(trainObj, te_set[20:21, :, notes], file_path=f"./output/test", instrument=choosedInstrument)
print("\n--------------------------------------------------\n")
generateAndSaveASong(trainObj, te_set[99:100, :, notes], file_path=f"./output/test2", instrument=choosedInstrument)
print("\n--------------------------------------------------\n")
#generateAndSaveASong(trainObj, file_path=f"./output/firstOriginalSong", instrument=choosedInstrument)

