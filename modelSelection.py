import itertools
from vaeModel import *
from train import *
import os

def modelSelection(tr_set, val_set, model, loss_function=loss_function_VAE, num_epochs=100, patience=20, directory_models=directory_models_RNN):


    print(f"\n---------------{Fore.BLUE}START MODEL SELECTION{Style.RESET_ALL}-------------------")
    input_size = tr_set.shape[2]

    for i in seeds:
        index = 0
        torch.manual_seed(i)
        print(f"MODEL SELECTION WITH SEED={i}")
        for params in itertools.product(latent_size_values, hidden_size_values, learning_rate_values, batch_size_values, num_layer_values, beta_values):
            print(f"{Fore.GREEN}Model Selection on Model {index} with parameters: {params} and seed={i}{Style.RESET_ALL}")
            latent_size, hidden_size, learning_rate, batch_size, num_layers, beta = params
            if exists(directory_models + f"/Seed={i}" + f"/{params}"):
                print("Already trained model: " + f"/Seed={i}" + f"/{params}")
                trainObj = loadVariableFromFile(directory_models + f"/Seed={i}" + f"/{params}")
                print(trainObj)
            else:
                if exists(directory_models + f"/Seed={i}" + f"/{params}_tmp"):
                    trainObj = loadVariableFromFile(directory_models + f"/Seed={i}" + f"/{params}_tmp")
                else:
                    modelToTrain = model.parametersInitialization(input_dim=input_size, hidden_dim=hidden_size, latent_dim=latent_size, num_layers=num_layers)
                    trainObj = trainingModel(modelToTrain)
                trainObj.trainModel(tr_set, val_set, batch_size=batch_size, loss_function=loss_function, num_epochs=num_epochs, patience=patience, learning_rate=learning_rate, beta=beta, saveModel=True, file_path=directory_models + f"/Seed={i}" + f"/{params}_tmp")
                saveVariableInFile(directory_models + f"/Seed={i}" + f"/{params}", trainObj)
                os.remove(directory_models + f"/Seed={i}" + f"/{params}_tmp")

            index += 1


    bestModelLoss, bestModelRecon = returnBestModels(directory_models, seeds)

    return bestModelLoss, bestModelRecon

def returnBestModels(directory, seeds):
    dictionaryModels = {}
    print(directory)
    for i in seeds:
        print(i)
        for filenames in os.listdir(directory + f"/Seed={i}"):
            print(filenames)
            model = loadVariableFromFile(directory + f"/Seed={i}/" + filenames)
            if filenames in dictionaryModels:
                dictionaryModels[filenames][0] += model.bestValLoss
                dictionaryModels[filenames][1] += model.valReconList[model.bestEpoch]
            else:
                dictionaryModels[filenames] = (model.bestValLoss, model.valReconList[model.bestEpoch])
            print(dictionaryModels[filenames])
    bestLossModel  = min(dictionaryModels.items(), key=lambda tup: tup[1][0] / len(seeds))
    bestReconModel = min(dictionaryModels.items(), key=lambda tup: tup[1][1] / len(seeds))

    return bestLossModel, bestReconModel
