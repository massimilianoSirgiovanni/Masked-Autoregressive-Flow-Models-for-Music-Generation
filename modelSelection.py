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
                trainObj = loadModel(directory_models + f"/Seed={i}" + f"/{params}", device=choosedDevice)
                print(trainObj)
            else:
                model = classModel()
                model.parametersInitialization((tr_set.tensors[0].shape[2], tr_set.tensors[0].shape[1]), hidden_sizes=hidden_sizes, madeType=madeType, n_layers=num_layers, num_genres=len(choosedGenres))
                trainObj = trainingModel(model).to(choosedDevice)
            trainObj.trainModel(tr_set, val_set, batch_size=batch_size, num_epochs=num_epochs, patience=patience, learning_rate=learning_rate, file_path=directory_models + f"/Seed={i}" + f"/{params}")
            saveModel(trainObj, directory_models + f"/Seed={i}" + f"/{params}")

            index += 1


    bestModel_label = returnBestModels(directory_models, seeds)
    choosedModel = loadModel(directory_models + f"/Seed={seeds[0]}" + f"/{bestModel_label}", device=choosedDevice)
    saveVariableInFile(directory_models + f"/bestModel_label_{madeType}", bestModel_label)

    return choosedModel, bestModel_label

def returnBestModels(directory, seeds):
    dictionaryModels = {}
    for i in seeds:
        for filenames in listdir(directory + f"/Seed={i}"):
            model = loadModel(directory + f"/Seed={i}/" + filenames, device=choosedDevice)
            if filenames in dictionaryModels:
                dictionaryModels[filenames] += model.bestValLoss
            else:
                dictionaryModels[filenames] = model.bestValLoss
    bestLossModel = min(dictionaryModels, key=lambda k: dictionaryModels[k].item()/len(seeds))
    bestLoss = dictionaryModels[bestLossModel]
    print(f"The Best Model is: {bestLossModel}, with loss={bestLoss}")
    return bestLossModel
