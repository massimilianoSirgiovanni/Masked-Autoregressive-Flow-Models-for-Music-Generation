import os
from colorama import Fore
from manageFiles import *
from torch import unique, LongTensor

def countGenresInTensor(tensor, percentage=False):
    n = tensor.shape[0]
    values, counts = unique(tensor, return_counts=True)
    for value, count in zip(values, counts):
        if percentage == False:
            print(f"Value: {value}, Frequency: {count}")
        else:
            print(f"Value: {value}, Frequency: {(count/n*100):.2f}%")

def extractGenre(file_name):
    return file_name.lower()[8:-4]

def getDictGenre(directory):
    if os.path.exists("./savedObjects/datasets/dictGenres"):
        dictGenres = loadVariableFromFile("./savedObjects/datasets/dictGenres")
        return dictGenres
    else:
        dictGenres = {}
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for filename in files:
        genre = extractGenre(filename)
        with open(f"{directory}/{filename}", 'r') as file:
            file_content = file.read()
            dictGenres[genre] = file_content.split('\n')
    saveVariableInFile("./savedObjects/datasets/dictGenres", dictGenres)
    return dictGenres

def selectGenre(genres, choosedGenres=list(getDictGenre("./amg").keys())):
    genre = genres[0]
    if genre in choosedGenres:
        return genre
    else:
        return None





def getGenreFromId(file_name, dictGenre=None):
    if dictGenre is None:
        dictGenre = getDictGenre("./amg")
    for i in dictGenre:
        if file_name in dictGenre[i]:
            return i
    return None

def gennreLabelToTensor(list_genres, choosedGenres):
    genre_to_int = {label: idx for idx, label in enumerate(choosedGenres)}
    genre_tensor = LongTensor([genre_to_int[label] for label in list_genres])
    return genre_tensor

def convertGenreToNumber(genre, choosedGenres=list(getDictGenre("./amg").keys())):
    if type(genre) == str:
        genre = genre.lower()
        for i in range(0, len(choosedGenres)):
            if choosedGenres[i] == genre:
                return i
        print(f"{Fore.RED}WARNING: The label \"{genre}\" provided does not correspond to any of the genres present")
        exit(-1)
    else:
        return genre

def convertGenreToString(genre, choosedGenres=list(getDictGenre("./amg").keys())):
    if type(genre) != str:
        return choosedGenres[int(genre)]
    else:
        return genre

def separateGenreToDataset(dataset):
    data = dataset[:, :, :-1]
    genres = dataset[:, 0, -1]
    return data, genres

