from sys import exit

from initFolders import *
import plot
from modelSelection import *
from mafModel import *
from genreAnalysis import *
from config import *
from generationFunct import *

from torchvision import datasets, transforms

import torch

def estimateDatasetsSize():
    for instrument in range(0, 128):
        for i in range(ord("A"), ord("A") + 1):
            if os.path.isfile(f"./savedObjects/datasets/2_bar/{chr(i)}/dataset_program={instrument}"):
                input = loadVariableFromFile(f"./savedObjects/datasets/2_bar/{chr(i)}/dataset_program={instrument}")
                if input != None:
                    print(input[0].shape)
                else:
                    print(input)


extendDataset = 0
torch.manual_seed(seeds[0])
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
#choosedModel.to(device)

notes = range(0, 130)
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
    print(f"Dataset Genre Frequency {countGenresInTensor(dataset.to_dense())}")
    '''dataset = dataset[genres == 1]
    genres = genres[genres == 1]
    genres[:] = 0'''
    print(dataset.shape)
    print(genres.shape)
    print(countGenresInTensor(genres))
    tr_set, val_set, te_set = holdoutSplit(dataset, genres, notes, val_percentage=0.3, test_percentage=0.3)
    accuracyMetrics.completeAnalisysOnSongsSets(tr_set.tensors[0].to_dense())
    del dataset, genres
    print(f"Training Genre Frequency:")
    countGenresInTensor(tr_set.tensors[1].to_dense())
    print(f"Validation Genre Frequency:")
    print(countGenresInTensor(val_set.tensors[1].to_dense()))
    print(f"Testing Genre Frequency:")
    print(countGenresInTensor(te_set.tensors[1].to_dense()))
    print(tr_set.tensors[0].shape)
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
        #genres = gennreLabelToTensor(genres, choosedGenres)
        tr_set, val_set, te_set = holdoutSplit(dataset, genres, notes, val_percentage=0.3, test_percentage=0.3)
        del dataset, genres
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



###################################################################################################################

if exists("./savedObjects/models/MAF_model"):
    trainObj = loadVariableFromFile("./savedObjects/models/MAF_model")
    print(trainObj)
else:
    #[(60, 50), (90, 80)]
    #[50, 80]
    #num_genres=len(choosedGenres),
    choosedModel.parametersInitialization((len(notes), 32), hidden_sizes=[50, 100, 80], n_layers=4, num_genres=len(choosedGenres), device=device, madeType='multinotes', activationLayer="relu")
    print(choosedModel)
    trainObj = trainingModel(choosedModel)
#plot.plotPairwiseGenres(tr_set.tensors[0].to_dense(), tr_set.tensors[1].to_dense(), 'pop_rock', second_genre=None, max_samples=250, seed=0)
#plot.plotPairwiseGenres(tr_set.tensors[0].to_dense(), tr_set.tensors[1].to_dense(), 'pop_rock', second_genre=1, max_samples=250, seed=0)
#plot.plotPairwiseGenres(tr_set.tensors[0].to_dense(), tr_set.tensors[1].to_dense(), 8, second_genre=4, max_samples=250, seed=0)

for i in range(0, 13):
    accuracyMetrics.completeAnalisysOnSingleGenre(tr_set.tensors[0].to_dense(), tr_set.tensors[1].to_dense(), i)

'''trainObj.testModel(tr_set, set_name="Training Set", batch_size=100)

tr_set = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transforms.ToTensor())
te_set = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transforms.ToTensor())
print(tr_set)

choosedModel = MAF()
choosedModel.parametersInitialization((1, 28*28), hidden_sizes=[50], n_layers=1, num_genres=1, device=device, madeType='univariate')
print(choosedModel)
trainObj = trainingModel(choosedModel)

trainObj.testMNIST(tr_set, set_name="Training Set", batch_size=60000)
trainObj.testMNIST(te_set, set_name="Test Set", batch_size=10000)
exit(0)'''
train = input("Do you want to train the model? (y/n): ")
if train.lower() == "y":
    genNoTrain = generateAndSaveASong(trainObj, file_path=f"./output/Epoch={0}", instrument=choosedInstrument, seed=0)
    plot.plot_piano_roll(genNoTrain, file_path=f"./output/Epoch={0}")
    #accuracyMetrics.completeAnalisysOnSongsSets(genNoTrain, "Epoch=0")
    trainObj.trainModel(tr_set, val_set, te_set, batch_size=100, loss_function=loss_function, num_epochs=100, patience=10, learning_rate=0.001, file_path="./savedObjects/models/MAF_model")

trainObj.plot_loss()
trainObj.plot_accuracy()

#trainObj.testModel(tr_set, set_name="Training Set", batch_size=tr_set.tensors[0].shape[0])
#trainObj.testModel(te_set, set_name="Test Set", batch_size=te_set.tensors[0].shape[0])

print("-"*8 + "GENERATION" + "-"*8)
print({label: idx for idx, label in enumerate(choosedGenres)})
print(tr_set.tensors[1][100].to_dense().shape)


def extractSong(set, number):
    return (set.tensors[0][number].to_dense().unsqueeze(0), set.tensors[1][number].to_dense())

song1 = extractSong(tr_set, 100)
print(song1)
song2 = extractSong(tr_set, 12)
output = generateAndSaveASong(trainObj.bestModel, song1, genres=song1[1], file_path=f"./output/trSetSong1", instrument=choosedInstrument)
print(accuracyMetrics.f1_score_with_flatten(output, song1[0]))
accuracyMetrics.completeAnalisysOnSongsSets(song1[0])
accuracyMetrics.completeAnalisysOnSongsSets(output)
plot.plot_piano_roll(output)

print("\n--------------------------------------------------\n")
output = generateAndSaveASong(trainObj.bestModel, song1, genres=torch.Tensor([5]), file_path=f"./output/trSetSong1Jazz", instrument=choosedInstrument)
print(accuracyMetrics.f1_score_with_flatten(output, song1[0]))
accuracyMetrics.completeAnalisysOnSongsSets(output)
plot.plot_piano_roll(output)

print("\n--------------------------------------------------\n")

output = generateAndSaveASong(trainObj.bestModel, song1, genres=torch.Tensor([10]), file_path=f"./output/trSetSong1Reggae", instrument=choosedInstrument)
print(accuracyMetrics.f1_score_with_flatten(output, song1[0]))
accuracyMetrics.completeAnalisysOnSongsSets(output)
plot.plot_piano_roll(output)

print("\n--------------------------------------------------\n")
exit(2)


output = generateAndSaveASong(trainObj.bestModel, song2, genres=song2[1], file_path=f"./output/trSetSong2", instrument=choosedInstrument)
print(accuracyMetrics.f1_score_with_flatten(output, song2[0]))
print("\n--------------------------------------------------\n")

interpolationFactor = 1.0
u = latentSpaceInterpolation(trainObj.bestModel, song1, song2, interpolationFactor=interpolationFactor)
output = generateFromLatentSpace(trainObj.bestModel, u, file_path=f"./output/interpolation={interpolationFactor}", instrument=choosedInstrument)
accuracyMetrics.completeAnalisysOnSongsSets(song1[0])
accuracyMetrics.completeAnalisysOnSongsSets(song2[0])
accuracyMetrics.completeAnalisysOnSongsSets(output)
print("\n--------------------------------------------------\n")
exit()

generateAndSaveASong(trainObj, file_path=f"./output/originalSong", instrument=choosedInstrument)
print("\n--------------------------------------------------\n")
exit()
generateAndSaveASong(trainObj, extractSong(tr_set, 30)[0], genres=extractSong(tr_set, 30)[1], file_path=f"./output/trSetCountry", instrument=choosedInstrument)
print("\n--------------------------------------------------\n")
exit(0)
generateAndSaveASong(trainObj, tr_set[60:61, :, notes], file_path=f"./output/trSet2", instrument=choosedInstrument)
print("\n--------------------------------------------------\n")
print("Test")
generateAndSaveASong(trainObj, te_set[20:21, :, notes], file_path=f"./output/test", instrument=choosedInstrument)
print("\n--------------------------------------------------\n")
generateAndSaveASong(trainObj, te_set[99:100, :, notes], file_path=f"./output/test2", instrument=choosedInstrument)
print("\n--------------------------------------------------\n")
#

