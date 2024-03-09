import pickle
from torch import save, load

def saveModel(model, file_path):
    save(model, file_path)
    print(f"SAVED MODEL IN: \"{file_path}\" ")

def loadModel(file_path, device):
    model = load(file_path, map_location=device)
    print(f"LOADED MODEL FROM: \"{file_path}\"")
    return model


# Funzione per il salvataggio dell'esecuzione in un file:
def saveVariableInFile(file_path, variable):
    picklefile = open(file_path, 'wb')
    pickle.dump(variable, picklefile)
    picklefile.close()
    print(f"SAVED VARIABLE IN: \"{file_path}\" ")


# Funzione per il caricamento dell'esecuzione da un file:
def loadVariableFromFile(file_path):
    picklefile = open(file_path, 'rb')
    variable = pickle.load(picklefile)
    picklefile.close()
    print(f"LOADED FILE FROM: \"{file_path}\"")
    return variable

def saveMIDI(midi_file, file_path):
    midi_file.write(file_path)
    print(f"MIDI Successfully Saved on: \"{file_path}\"")
