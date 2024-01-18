from os import makedirs
from os.path import exists
from initFolders import current_directory
from genreAnalysis import getDictGenre
from accuracyMetrics import f1_score_with_flatten


seeds = [0]

for i in seeds:
    if not exists(f"{current_directory}/savedObjects/models/RNN/Seed={i}"):
        makedirs(f"{current_directory}/savedObjects/models/RNN/Seed={i}")
    if not exists(f"{current_directory}/savedObjects/models/LSTM/Seed={i}"):
        makedirs(f"{current_directory}/savedObjects/models/LSTM/Seed={i}")



choosedInstrument = 2

choosedGenres = list(getDictGenre("./amg").keys())

accuracyFunct = f1_score_with_flatten

latent_size_values = [16, 32, 64]
hidden_size_values = [96, 128]
learning_rate_values = [0.001, 0.01]
batch_size_values = [1000, 1500]
num_layer_values = [2, 3]
beta_values = [0.1]


