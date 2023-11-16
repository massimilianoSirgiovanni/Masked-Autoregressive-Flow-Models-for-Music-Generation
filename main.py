import torch

from manageMIDI import *
from sys import exit
from models import *
from train import *
from modelSelection import *
from flowBasedModels import *

extendDataset = 0
torch.manual_seed(seeds[0])
torch.autograd.detect_anomaly(True)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f"Device: {device}")
print(f"program {choosedInstrument}")

print("--------------------Starting Execution---------------------")
print(f"{Fore.YELLOW}Seed = {seeds[0]}{Style.RESET_ALL}\n")

print(f"Choose Model:\n{Fore.LIGHTGREEN_EX}1. RNN;\n2. LSTM;\n3. MAF;{Style.RESET_ALL}\n")
value = input("Enter your choice here: ")
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
value = input("Enter your choice here: ")

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
    forward = input(f"{Fore.MAGENTA}Do you want to continue running the Model Selection? (y/n) {Style.RESET_ALL}")
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
    choosedModel.parametersInitialization(tr_set.shape[2], tr_set.shape[1], n_layers=1, hidden_sizes=[500], device=device)
    print(choosedModel)
    trainObj = trainingModel(choosedModel)
#trainObj.trainModel(tr_set, val_set, te_set, batch_size=500, loss_function=loss_function, num_epochs=50, patience=10, learning_rate=0.5, file_path="./savedObjects/models/MAF_model")

trainObj.plot_loss()
displaySong(trainObj, te_set, 0)
displaySong(trainObj, te_set,  10)
displaySong(trainObj, te_set, 100)
exit(0)
#########################################################################################

saved = input("Saved")
if saved == "1":
    trainObj = loadVariableFromFile("./savedObjects/models/bestReconModel=(32, 96, 0.01, 1000, 3, 0.1)")
else:
    model = VAE(nn.LSTM, input_dim=tr_set.shape[2], hidden_dim=96, latent_dim=32, num_layers=3, cell=True)
    trainObj = trainingModel(model)
print(trainObj)

train = input("Si allena? ")
if train == "1":

    displaySong(trainObj, 0)
    trainObj.trainModel(tr_set, val_set, batch_size=1000, loss_function=loss_function_VAE, num_epochs=100, patience=20, learning_rate=0.01, file_path="./savedObjects/models/bestReconModel=(32, 96, 0.01, 1000, 3, 0.1)")

displaySong(trainObj, 0)
displaySong(trainObj, 10)
displaySong(trainObj, 100)

exit(0)


'''dataset = loadVariableFromFile("./savedObjects/datasets/2_bar/dataset_complete_program=0")
tr_set, val_set, te_set = holdoutSplit(dataset, val_percentage=0.2, test_percentage=0.2)

train = loadVariableFromFile(directory_models_LSTM + "/Seed=121/(32, 128, 0.01, 1000, 3, 0.1)")
train.plot_loss()
#print(returnBestModels(directory_models_LSTM, seeds))
#4, 10, 100, 200
displaySong(train, 38)
exit(0)'''

#///////////////////////////////////////////////////////////////////////////////////////////////7
#print(loadLMD("./lmd_matched", starting_point='A', ending_point='Z'))

#dataset = getSingleInstrumentDatabaseLMD(f"./savedObjects/datasets/2_bar", 0)
dataset = loadVariableFromFile("./savedObjects/datasets/2_bar/dataset_complete_program=0")
print(dataset.shape)
tr_set, val_set, te_set = holdoutSplit(dataset, val_percentage=0.2, test_percentage=0.2)
print(f"Training Set Size: {tr_set.shape}")
print(f"Validation Set Size: {val_set.shape}")

model = VAE(nn.LSTM, input_dim=tr_set.shape[2], hidden_dim=96, latent_dim=32, num_layers=3, cell=True)
trainObj = trainingModel(model)
print(trainObj)
trainObj.trainModel(tr_set, val_set, batch_size=1000, loss_function=loss_function_VAE, num_epochs=100, patience=10, learning_rate=0.01, file_path="./savedObjects/models/bestReconModel=(32, 96, 0.01, 1000, 3, 0.1)")
exit(0)

print(modelSelection(tr_set, val_set, choosedModel, directory_models=choosedDirectory, patience=10))






