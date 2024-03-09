from os import makedirs, getcwd
from os.path import exists

current_directory = getcwd()

if not exists(f"{current_directory}/savedObjects"):
    makedirs(f"{current_directory}/savedObjects")

if not exists(f"{current_directory}/savedObjects/datasets"):
    makedirs(f"{current_directory}/savedObjects/datasets")

if not exists(f"{current_directory}/savedObjects/models"):
    makedirs(f"{current_directory}/savedObjects/models")

if not exists(f"{current_directory}/savedObjects/models/RNN"):
    makedirs(f"{current_directory}/savedObjects/models/RNN")

if not exists(f"{current_directory}/savedObjects/models/LSTM"):
    makedirs(f"{current_directory}/savedObjects/models/LSTM")

if not exists(f"{current_directory}/output"):
    makedirs(f"{current_directory}/output")

if not exists(f"{current_directory}/output/LSI"):
    makedirs(f"{current_directory}/output/LSI")

if not exists(f"{current_directory}/output/Conditioning"):
    makedirs(f"{current_directory}/output/Conditioning")

directory_dataset = f"{current_directory}/savedObjects/datasets/2_bar"
directory_models_RNN = f"{current_directory}/savedObjects/models/RNN"
directory_models_LSTM = f"{current_directory}/savedObjects/models/LSTM"
directory_models_MAF_Shared = f"{current_directory}/savedObjects/models/MAF/Shared"
directory_models_MAF_DNDW = f"{current_directory}/savedObjects/models/MAF/DNDW"
directory_models_MAF_Multivariate = f"{current_directory}/savedObjects/models/MAF/Multivariate"
