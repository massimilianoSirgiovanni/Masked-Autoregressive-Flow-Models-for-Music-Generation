import pickle


# Funzione per il salvataggio dell'esecuzione in un file:
def saveVariableInFile(file_path, variable):
    picklefile = open(file_path, 'wb')
    pickle.dump(variable, picklefile)
    picklefile.close()
    print(f"SAVED VARIABLE: \"{variable}\" ")


# Funzione per il caricamento dell'esecuzione da un file:
def loadVariableFromFile(file_path):
    picklefile = open(file_path, 'rb')
    variable = pickle.load(picklefile)
    picklefile.close()
    print(f"Caricamento del file: {file_path}")
    return variable

def saveMIDI(midi_file, file_path):
    midi_file.write(file_path)
    print("MIDI Successfully Saved!")