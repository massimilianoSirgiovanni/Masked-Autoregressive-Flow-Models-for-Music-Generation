import pickle


# Funzione per il salvataggio dell'esecuzione in un file:
def saveVariableInFile(file_path, variable):
    picklefile = open(file_path, 'wb')
    pickle.dump(variable, picklefile)
    picklefile.close()


# Funzione per il caricamento dell'esecuzione da un file:
def loadVariableFromFile(file_path):
    picklefile = open(file_path, 'rb')
    variable = pickle.load(picklefile)
    picklefile.close()
    print(f"Caricamento del file: {file_path}")
    return variable