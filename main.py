import mido
from manageMIDI import *
from manageFiles import *
from sys import exit

import numpy as np

import music21

# Lettura il file MIDI
midi_file = mido.MidiFile('./Dataset MIDI/Full Metal Alchemist_ Brotherhood - Lurking.mid')


#print(midi_file)

tensor_midi = loadVariableFromFile("./variables/midi_tensor")
print(tensor_midi[1:5, :])
print("Tensore")
print(tensor_midi.shape)

# Variabili per la rappresentazione dei dati MIDI
max_time_steps = 10#0  # Numero massimo di passi temporali
num_pitches = 128      # Numero di possibili altezze musicali

if os.path.isfile("./variables/dataset"):
    dataset = loadVariableFromFile("./variables/dataset")
else:
    dataset = loadDataset("./Dataset MIDI", max_time_steps, num_pitches)
    saveVariableInFile("./variables/dataset", dataset)

print(dataset.shape)

exit(0)