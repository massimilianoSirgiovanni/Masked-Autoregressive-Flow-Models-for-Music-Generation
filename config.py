from os import makedirs, getcwd
from os.path import exists
import random
from colorama import Fore, Style
import torch.nn as nn

current_directory = getcwd()

choosedInstrument = 1

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

seeds = [0]

for i in seeds:
    if not exists(f"{current_directory}/savedObjects/models/RNN/Seed={i}"):
        makedirs(f"{current_directory}/savedObjects/models/RNN/Seed={i}")
    if not exists(f"{current_directory}/savedObjects/models/LSTM/Seed={i}"):
        makedirs(f"{current_directory}/savedObjects/models/LSTM/Seed={i}")

directory_dataset = f"{current_directory}/savedObjects/datasets/2_bar"
directory_models_RNN = f"{current_directory}/savedObjects/models/RNN"
directory_models_LSTM = f"{current_directory}/savedObjects/models/LSTM"
directory_models_MAF = f"{current_directory}/savedObjects/models/MAF"




latent_size_values = [16, 32, 64]
hidden_size_values = [96, 128]
learning_rate_values = [0.001, 0.01]
batch_size_values = [1000, 1500]
num_layer_values = [2, 3]
beta_values = [0.1]


