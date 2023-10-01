from manageMIDI import *
from sys import exit
from models import *

dataset = loadDataset("./variables")
dictionary = {}
for i in dataset:
    for instrument in i.instruments:
        piano_roll = midi_to_piano_roll(dataset[0], instruments=[instrument.program])
        dictionary = piano_roll_to_dictionary(dictionary, piano_roll, instrument.program)
print("Piano Roll obtained!")
midi_data = piano_roll_to_midi(dictionary)

print(f"Instruments {midi_data.instruments}")

saveMIDI(midi_data, "./output/new_midi.mid")

exit(0)

if os.path.isfile("./variables/dataset"):
    dataset = loadVariableFromFile("./variables/dataset")
else:
    dataset = loadLMD("./lmd_matched", starting_point="A", ending_point="B")
    saveVariableInFile("./variables/dataset", dataset)
print(dataset.shape)
exit(0)
