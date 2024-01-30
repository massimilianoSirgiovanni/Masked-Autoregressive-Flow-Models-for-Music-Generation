from os import makedirs
from os.path import exists
from initFolders import current_directory
from genreAnalysis import getDictGenre
from accuracyMetrics import f1_score_with_flatten
from torch import device, cuda


seeds = [0]

for i in seeds:
    if not exists(f"{current_directory}/savedObjects/models/MAF/Shared/Seed={i}"):
        makedirs(f"{current_directory}/savedObjects/models/MAF/Shared/Seed={i}")
    if not exists(f"{current_directory}/savedObjects/models/MAF/DNDW/Seed={i}"):
        makedirs(f"{current_directory}/savedObjects/models/MAF/DNDW/Seed={i}")
    if not exists(f"{current_directory}/savedObjects/models/MAF/Multivariate/Seed={i}"):
         makedirs(f"{current_directory}/savedObjects/models/MAF/Multivariate/Seed={i}")



choosedInstrument = 0

# Parameters Configuration
choosedDevice = device('cuda' if cuda.is_available() else 'cpu')

val_percentage = 0.3
test_percentage = 0.2

choosedGenres = list(getDictGenre("./amg").keys())

accuracyFunct = f1_score_with_flatten

hidden_sizes_shared = [[500, 1000]]#[[100, 200], [200, 500], [500, 1000]]
hidden_sizes_dndw = [[50, 100], [100, 200]]
hidden_sizes_multivariate = [[(10, 10), (15, 10)]] #[[(10, 5), (10, 10)], [(10, 10), (20, 10)]]
num_layers_values = [4]#[2, 3, 4]
learning_rate_values = [0.001]
batch_size_values = [1000]


