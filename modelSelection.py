from itertools import product
from train import *
from os import remove, listdir
from torch import manual_seed

def modelSelectionMAF(tr_set, val_set, classModel, madeType='shared', num_epochs=100, patience=20, directory_models="./savedObjects/models"):


    print(f"\n---------------{Fore.BLUE}START MODEL SELECTION{Style.RESET_ALL}-------------------")

    for i in seeds:
        index = 0
        manual_seed(i)
        print(f"MODEL SELECTION WITH SEED={i}")
        if madeType == 'shared':
            hidden_sizes_values = hidden_sizes_shared
        elif madeType == 'dndw':
            hidden_sizes_values = hidden_sizes_dndw
        elif madeType == 'multivariate':
            hidden_sizes_values = hidden_sizes_multivariate
        for params in product(hidden_sizes_values, num_layers_values, batch_size_values, learning_rate_values):
            print(f"{Fore.GREEN}Model Selection on Model {index} (madeType = {madeType}) with parameters: {params} and seed={i}{Style.RESET_ALL}")
            hidden_sizes, num_layers, batch_size, learning_rate = params
            if exists(directory_models + f"/Seed={i}" + f"/{params}"):
                print("Already trained model: " + f"/Seed={i}" + f"/{params}")
                trainObj = loadModel(directory_models + f"/Seed={i}" + f"/{params}", device=choosedDevice)
                print(trainObj)
            else:
                if exists(directory_models + f"/Seed={i}" + f"/{params}_tmp"):
                    trainObj = loadModel(directory_models + f"/Seed={i}" + f"/{params}_tmp", device=choosedDevice)
                else:
                    model = classModel()
                    model.parametersInitialization((tr_set.tensors[0].shape[2], tr_set.tensors[0].shape[1]), hidden_sizes=hidden_sizes, madeType=madeType, n_layers=num_layers, num_genres=len(choosedGenres))
                    trainObj = trainingModel(model).to(choosedDevice)
                trainObj.trainModel(tr_set, val_set, batch_size=batch_size, num_epochs=num_epochs, patience=patience, learning_rate=learning_rate, file_path=directory_models + f"/Seed={i}" + f"/{params}_tmp")
                saveModel(trainObj, directory_models + f"/Seed={i}" + f"/{params}")
                remove(directory_models + f"/Seed={i}" + f"/{params}_tmp")

            index += 1


    bestModel = returnBestModels(directory_models, seeds)

    return bestModel

def returnBestModels(directory, seeds):
    dictionaryModels = {}
    print(directory)
    for i in seeds:
        print(i)
        for filenames in listdir(directory + f"/Seed={i}"):
            print(filenames)
            model = loadVariableFromFile(directory + f"/Seed={i}/" + filenames)
            if filenames in dictionaryModels:
                dictionaryModels[filenames] += model.bestValLoss
            else:
                dictionaryModels[filenames] = model.bestValLoss
            print(dictionaryModels[filenames])
    bestLossModel  = min(dictionaryModels.items(), key=lambda tup: tup[1][0] / len(seeds))

    return bestLossModel
