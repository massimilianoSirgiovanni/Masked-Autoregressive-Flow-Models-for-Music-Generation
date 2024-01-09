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

directory_dataset = f"{current_directory}/savedObjects/datasets/2_bar"
directory_models_RNN = f"{current_directory}/savedObjects/models/RNN"
directory_models_LSTM = f"{current_directory}/savedObjects/models/LSTM"
directory_models_MAF = f"{current_directory}/savedObjects/models/MAF"
