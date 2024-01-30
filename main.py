from sys import exit

from initFolders import *
from modelSelection import *
from mafModel import *
from genreAnalysis import *
from config import *
from generationFunct import *
from torch import cuda, manual_seed


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
print(f"program {choosedInstrument}")
print(choosedGenres)
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
elif value == "1":
    print(f"{Fore.LIGHTGREEN_EX}-------------PREPROCESSING AND ANALYSIS------------------{Style.RESET_ALL}")
    dataset, genres = getSingleInstrumentDatabaseLMD(f"./savedObjects/datasets/2_bar", choosedInstrument)
    print(f"Dataset dim: {dataset.shape}")
    print(f"Dataset Occurrence Frequency:")
    countGenresInTensor(dataset.to_dense())
    print()
    print(f"Genre labels dim: {genres.shape}")
    print("Genre Frequencies in Dataset")
    countGenresInTensor(genres)
    tr_set, val_set, te_set = holdoutSplit(dataset, genres, val_percentage=val_percentage, test_percentage=test_percentage)
    accuracyMetrics.completeAnalisysOnSongsSets(tr_set.tensors[0].to_dense(), stringGenre="TRAINING")
    del dataset, genres
    print()
    print(f"{Fore.MAGENTA}Training Genre Frequency:{Style.RESET_ALL}")
    countGenresInTensor(tr_set.tensors[1].to_dense())
    print()
    print(f"{Fore.MAGENTA}Validation Genre Frequency:{Style.RESET_ALL}")
    countGenresInTensor(val_set.tensors[1].to_dense())
    print()
    print(f"{Fore.MAGENTA}Testing Genre Frequency:{Style.RESET_ALL}")
    countGenresInTensor(te_set.tensors[1].to_dense())
    print()
    forward = input(f"{Fore.MAGENTA}Do you want to continue running the Model Selection? (y/n) {Style.RESET_ALL}")
    if forward.lower() == "y":
        value = "2"
    else:
        pass
        #exit(0)
elif value == "2":
    print(f"{Fore.LIGHTGREEN_EX}------------------MODEL SELECTION------------------{Style.RESET_ALL}")
    if tr_set == None or val_set  == None:
        dataset, genres = getSingleInstrumentDatabaseLMD(f"./savedObjects/datasets/2_bar", choosedInstrument)
        tr_set, val_set, te_set = holdoutSplit(dataset, genres, val_percentage=val_percentage, test_percentage=test_percentage)
        del dataset, genres
    modelSelectionMAF(tr_set, val_set, choosedModel, madeType=madeType, num_epochs=500, directory_models=choosedDirectory, patience=30)
    forward = input(f"{Fore.MAGENTA}Do you want to continue running the Training? (y/n) {Style.RESET_ALL}")
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

#plot.plotPairwiseGenres(tr_set.tensors[0].to_dense(), tr_set.tensors[1].to_dense(), 'pop_rock', second_genre=None, max_samples=250, seed=0)
#plot.plotPairwiseGenres(tr_set.tensors[0].to_dense(), tr_set.tensors[1].to_dense(), 'pop_rock', second_genre=1, max_samples=250, seed=0)
#plot.plotPairwiseGenres(tr_set.tensors[0].to_dense(), tr_set.tensors[1].to_dense(), 8, second_genre=4, max_samples=250, seed=0)



'''print("-"*8 + "GENERATION" + "-"*8)
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
plot.plot_piano_roll(output, file_path=f"./output/trSetSong1")

print("\n--------------------------------------------------\n")
output = generateAndSaveASong(trainObj.bestModel, song1, genres=torch.Tensor([1]), file_path=f"./output/trSetSong1Country", instrument=choosedInstrument)
print(accuracyMetrics.f1_score_with_flatten(output, song1[0]))
accuracyMetrics.completeAnalisysOnSongsSets(output)
plot.plot_piano_roll(output)
plot.plot_piano_roll(output, file_path=f"./output/trSetSong1Country")

print("\n--------------------------------------------------\n")

output = generateAndSaveASong(trainObj.bestModel, song1, genres=torch.Tensor([11]), file_path=f"./output/trSetSong1Electronic", instrument=choosedInstrument)
print(accuracyMetrics.f1_score_with_flatten(output, song1[0]))
accuracyMetrics.completeAnalisysOnSongsSets(output)
plot.plot_piano_roll(output)
plot.plot_piano_roll(output, file_path=f"./output/trSetSong1Electronic")

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
print("\n--------------------------------------------------\n")'''


