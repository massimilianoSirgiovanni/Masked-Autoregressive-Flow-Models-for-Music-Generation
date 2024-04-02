from sys import exit

import numpy.random
import torch

import accuracyMetrics
from initFolders import *
from modelSelection import *
from mafModel import *
from genreAnalysis import *
from config import *
from generationFunct import *
from torch import cuda, manual_seed, randint, tensor, multinomial
import plot
import random



def loadDatasetWithHoldout():
    dataset, genres = getSingleInstrumentDatabaseLMD(f"./savedObjects/datasets/2_bar", choosedInstrument)
    tr_set, val_set, te_set = holdoutSplit(dataset, genres, val_percentage=val_percentage, test_percentage=test_percentage)
    del dataset, genres
    return tr_set, val_set, te_set

def latentSpaceInterpolationTest(model, set, songID_1, songID_2, genre=Tensor([0]), directory="./output"):

    song1 = extractSong(set, songID_1)
    print(song1)
    song2 = extractSong(set, songID_2)
    print(song2)
    for a in [0, 0.3, 0.5, 0.7, 1]:
        z = latentSpaceInterpolation(model, (song1[0], genre), (song2[0], genre), interpolationFactor=a)
        output = generateFromLatentSpace(model, z, genre=genre, file_path=f"{directory}/LSI-factor={a}")
        plot.plot_piano_roll(output, file_path=f"{directory}/LSI-factor={a}")


loadOriginalDataset = False

if loadOriginalDataset:
    loadLPD("./lpd/lpd_cleansed")


extendDataset = 0
manual_seed(seeds[0])
if cuda.is_available():
    print(cuda.get_device_name(0))
else:
    print("CUDA not available on this system.")

print(f"Device: {choosedDevice}")
print(f"MIDI Program: {choosedInstrument}")
print(f"Genres: {choosedGenres}")
print("--------------------Starting Execution---------------------")
print(f"{Fore.YELLOW}Seed = {seeds[0]}{Style.RESET_ALL}\n")


print(f"Choose MAF:\n{Fore.LIGHTGREEN_EX}1. Shared Weight on Notes;\n2. Different Notes Different Weights;\n3. Multivariate;{Style.RESET_ALL}\n")
value = input("Enter your choice here: ")
if value not in ["1", "2", "3"]:
    print(f"{Fore.RED}ERROR: The entered value does not match any selectable options{Style.RESET_ALL}")
    exit(-1)
if value == "1":
    madeType = 'shared'
    choosedDirectory = directory_models_MAF_Shared
elif value == "2":
    madeType='dndw'
    choosedDirectory = directory_models_MAF_DNDW
elif value == "3":
    madeType='multivariate'
    choosedDirectory = directory_models_MAF_Multivariate

choosedModel = MAF

tr_set=None;val_set=None;test_set = None;dataset = None
print(f"Enter the corresponding number to choose what you want to execute:\n{Fore.LIGHTGREEN_EX}1. Download and preprocessing on data\n2. Model Selection\n3. Train the choosen model\n4. Testing the choosen model\n5. Complete Execution{Style.RESET_ALL}")
value = input("Enter your choice here: ")

if value not in ["1", "2", "3", "4", "5"]:
    print(f"{Fore.RED}ERROR: The entered value does not match any selectable options{Style.RESET_ALL}")
    exit(-1)
if value == "1" or value=='5':
    print(f"{Fore.LIGHTGREEN_EX}-------------PREPROCESSING AND ANALYSIS------------------{Style.RESET_ALL}")
    dataset, genres = getSingleInstrumentDatabaseLMD(f"./savedObjects/datasets/2_bar", choosedInstrument)
    print(f"Dataset dim: {dataset.shape}")
    #print(f"Dataset Occurrence Frequency:")
    #countGenresInTensor(dataset.to_dense())
    print()
    print(f"Genre labels dim: {genres.shape}")
    print("Genre Frequencies in Dataset")
    countGenresInTensor(genres, True)
    tr_set, val_set, te_set = holdoutSplit(dataset, genres, val_percentage=val_percentage, test_percentage=test_percentage)
    accuracyMetrics.completeAnalisysOnSongsSets(tr_set.tensors[0].to_dense(), stringGenre="TRAINING")
    del dataset, genres
    print()
    print(f"{Fore.MAGENTA}Training Genre Frequency:{Style.RESET_ALL}")
    countGenresInTensor(tr_set.tensors[1].to_dense(), True)
    print()
    print(f"{Fore.MAGENTA}Validation Genre Frequency:{Style.RESET_ALL}")
    countGenresInTensor(val_set.tensors[1].to_dense(), True)
    print()
    print(f"{Fore.MAGENTA}Testing Genre Frequency:{Style.RESET_ALL}")
    countGenresInTensor(te_set.tensors[1].to_dense(), True)
    print()
    if value != '5':
        forward = input(f"{Fore.MAGENTA}Do you want to continue running the Model Selection? (y/n) {Style.RESET_ALL}")
        if forward.lower() == "y":
            value = "2"
        else:
            exit(0)
if value == "2" or value=='5':
    print(f"{Fore.LIGHTGREEN_EX}------------------MODEL SELECTION------------------{Style.RESET_ALL}")
    if tr_set == None or val_set  == None:
        tr_set, val_set, te_set = loadDatasetWithHoldout()
    choosedModel, label = modelSelectionMAF(tr_set, val_set, choosedModel, madeType=madeType, num_epochs=500, directory_models=choosedDirectory, patience=model_selection_patience)
    saveModel(choosedModel, choosedDirectory + f"/{madeType}FinalModel={label}")
    forward = input(f"{Fore.MAGENTA}Do you want to continue running the Training? (y/n) {Style.RESET_ALL}")
    if value != '5':
        if forward.lower() == "y":
            value = "3"
        else:
            exit(0)
if value == "3" or value=='5':
    print(f"{Fore.LIGHTGREEN_EX}-------------TRAINING------------------{Style.RESET_ALL}")
    if not exists(choosedDirectory + f"/bestModel_label_{madeType}"):
        print(f"{Fore.RED}ERROR: You can not train the best model without executing Model Selection first{Style.RESET_ALL}")
        exit(-1)

    if tr_set == None or val_set  == None:
        tr_set, val_set, te_set = loadDatasetWithHoldout()
    if madeType == 'shared':
        hidden_sizes_values = hidden_sizes_shared
    elif madeType == 'dndw':
        hidden_sizes_values = hidden_sizes_dndw
    elif madeType == 'multivariate':
        hidden_sizes_values = hidden_sizes_multivariate
    choosedParams = loadVariableFromFile(choosedDirectory + f"/bestModel_label_{madeType}")
    file_name = f"/{madeType}FinalModel={choosedParams}"
    for params in product(hidden_sizes_values, num_layers_values, batch_size_values, learning_rate_values):
        if str(params) == choosedParams:
            hidden_sizes, num_layers, batch_size, learning_rate = params
            if exists(choosedDirectory + f"{file_name}"):
                trainObj = loadModel(choosedDirectory + f"{file_name}", device=choosedDevice)
                print(trainObj)
            else:
                model = choosedModel()
                model.parametersInitialization((tr_set.tensors[0].shape[2], tr_set.tensors[0].shape[1]), hidden_sizes=hidden_sizes, madeType=madeType, n_layers=num_layers, num_genres=len(choosedGenres))
                trainObj = trainingModel(model).to(choosedDevice)
            trainObj.trainModel(tr_set, val_set, batch_size=batch_size, num_epochs=500, patience=training_patience, learning_rate=learning_rate, file_path=choosedDirectory + f"{file_name}")
            saveModel(trainObj, choosedDirectory + file_name)
            plot.plot_loss(trainObj, file_path=f"./output/TrainingPlot-{madeType}")
            break
    if value != '5':
        forward = input(f"{Fore.MAGENTA}Do you want to continue running ? (y/n) {Style.RESET_ALL}")
        if forward.lower() == "y":
            value = "4"
        else:
            exit(0)
if value == "4" or value=='5':
    if tr_set == None or te_set  == None:
        tr_set, val_set, te_set = loadDatasetWithHoldout()

    print(f"{Fore.LIGHTGREEN_EX}-------------TESTING------------------{Style.RESET_ALL}")
    print(f"Enter the corresponding number to choose what you want to execute:\n{Fore.LIGHTGREEN_EX}1. Accuracy on Test Set\n2. Evaluate Generation on 1000 sample\n3. Latent Space Interpolation Example\n4. Generation Example\n5. Evaluate Conditioning on 1000 sample\n6. Conditioning Generation Example\n7. Complete Execution {Style.RESET_ALL}")
    value = input("Enter your choice here: ")
    choosedParams = loadVariableFromFile(choosedDirectory + f"/bestModel_label_{madeType}")
    file_name = f"/{madeType}FinalModel={choosedParams}"
    if exists(choosedDirectory + file_name):
        trainObj = loadModel(choosedDirectory + file_name, device=choosedDevice)
        print(trainObj)
    else:
        print(f"{Fore.RED}ERROR: No Final Model is saved, please try to rerun Model Selection and Training")
        exit(-1)
    if value == '1' or value=='7':
        print(5 * '-' + "Accuracy on Test Set" + 5 * '-')
        trainObj.testModel(te_set, set_name="Test Set", batch_size=200)
        if value != '7':
            forward = input(f"{Fore.MAGENTA}Do you want to continue running \"Evaluate Generation on 1000 sample\"? (y/n) {Style.RESET_ALL}")
            if forward.lower() == "y":
                value = "2"
            else:
                exit(0)

    elif value == '2' or value=='7':
        print(5*'-' + "Evaluate Generation on 1000 sample" + 5*'-')
        accuracyMetrics.completeAnalisysOnSongsSets(tr_set.tensors[0].to_dense(), stringGenre="TRAINING")
        n = 1000
        probabilities = tensor([0.005, 0.09, 0.096, 0.008, 0.031, 0.031, 0.063, 0.007, 0.565, 0.021, 0.005, 0.063, 0.015])
        genresRandom = multinomial(probabilities, n, replacement=True)
        generatedX = trainObj.generate(n_samples=n, genres=genresRandom, seed=0)
        accuracyMetrics.completeAnalisysOnSongsSets(generatedX, stringGenre=madeType)
        if value != '7':
            forward = input(f"{Fore.MAGENTA}Do you want to continue running \"Latent Space Interpolation Example\"? (y/n) {Style.RESET_ALL}")
            if forward.lower() == "y":
                value = "3"
            else:
                exit(0)

    elif value == '3' or value=='7':
        print(5 * '-' + "Latent Space Interpolation Example" + 5 * '-')
        latentSpaceInterpolationTest(trainObj.bestModel, te_set, 11, 0, genre=Tensor([8]), directory="./output/LSI")
        if value != '7':
            forward = input(f"{Fore.MAGENTA}Do you want to continue running \"Generation Examples\"? (y/n) {Style.RESET_ALL}")
            if forward.lower() == "y":
                value = "4"
            else:
                exit(0)

    elif value == '4' or value=='7':
        print(5 * '-' + "Generation Examples" + 5 * '-')
        for seed in choosedGenerateSeed:
            genres = Tensor([0.07, 0.07, 0.07, 0.07, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]).unsqueeze(0)
            song = generateAndSaveASong(model=trainObj, genres=genres, file_path=f"./output/newSong-{madeType}-Seed={seed}-", seed=seed)
            plot.plot_piano_roll(song, file_path=f"./output/newSong-{madeType}-Seed={seed}")
            if value != '7':
                forward = input(
                    f"{Fore.MAGENTA}Do you want to continue running \"Evaluate Conditioning on 1000 sample\"? (y/n) {Style.RESET_ALL}")
                if forward.lower() == "y":
                    value = "5"
                else:
                    exit(0)
    elif value == '5' or value=='7':
        print(5 * '-' + "Evaluate Conditioning on 1000 sample" + 5 * '-')
        for genre in range(0, len(choosedGenres)):
            accuracyMetrics.completeAnalisysOnSingleGenre(tr_set.tensors[0], tr_set.tensors[1], genres)
            n = 1000
            genres = Tensor([genre])
            genres = genres.repeat(n)
            conditionGeneratedX = trainObj.generate(n_samples=n, genres=genres, seed=0)
            accuracyMetrics.completeAnalisysOnSongsSets(conditionGeneratedX, stringGenre=f"{madeType} - {convertGenreToString(genre)}")

        if value != '7':
            forward = input(
                f"{Fore.MAGENTA}Do you want to continue running \"Conditioning Generation Example\"? (y/n) {Style.RESET_ALL}")
            if forward.lower() == "y":
                value = "6"
            else:
                exit(0)
    elif value == '6' or value=='7':
        print(5 * '-' + "Conditioning Generation Example" + 5 * '-')
        for genre in range(0, len(choosedGenres)):
            song = generateAndSaveASong(model=trainObj, genres=Tensor([genre]), file_path=f"./output/Conditioning/newSong-{madeType}-Genre={genre}-", seed=0)
            plot.plot_piano_roll(song, file_path=f"./output/Conditioning/newSong-{madeType}-Genre={genre}-")


