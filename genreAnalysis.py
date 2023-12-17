import h5py
import spotipy
import os
from spotipy.oauth2 import SpotifyClientCredentials
from manageFiles import *
from config import *

# Imposta le tue credenziali Spotify
client_id = '106578ff192d4d98a67ec0221310d21d'
client_secret = '28e3526272f447ccad6e238d2151206b'

# Inizializza l'oggetto Spotipy
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))


def getGenreFromSpotify(artist_name):

    result = sp.search(q=artist_name, type='artist', limit=1)
    return result['artists']['items'][0]['genres']


def getArtistName(file_path):
    with h5py.File(file_path, 'r') as file:
        return file['metadata']['songs']['artist_name'][0]

def getGenre(file_path):
    with h5py.File(file_path, 'r') as file:
        return [file['metadata']['songs']['genre'][0]]


def loadSongGenres(directory, dictGenres):
    print(f"Loading Dataset from {directory} ...")
    for dirpath, dirnames, filenames in os.walk(directory):
        print(dirpath)
        print(dictGenres)
        for filename in [f for f in filenames if f.endswith(".h5")]:
            try:
                #artistName = getArtistName(f"{dirpath}/{filename}")
                #genres = getGenre(artistName)
                genres = getGenre(f"{dirpath}/{filename}")
                print(genres)
                for genre in genres:
                    if genre not in dictGenres:
                        dictGenres[genre] = 1
                    else:
                        dictGenres[genre] += 1

            except Exception as e:
                print(f"Skipped cause of:\n{e}")
                pass

def countGenres(directory, starting_point="A", ending_point="Z"):
    if exists("./savedObjects/datasets/dictGenres"):
        dictGenres = loadVariableFromFile("./savedObjects/datasets/dictGenres")
        dictGenres = dict(sorted(dictGenres.items(), key=lambda item: item[1], reverse=True))
        print(f"Dict Len: {len(dictGenres)}")
        j = 0
        for i in dictGenres:
            j += 1
            print(f"{j}. {i}: {dictGenres[i]}")
        exit(0)
    else:
        dictGenres = {}
    for i in range(ord(starting_point), ord(ending_point) + 1):
        loadSongGenres(f"{current_directory}/{directory}/{chr(i)}", dictGenres)
        saveVariableInFile("./savedObjects/datasets/dictGenres", dictGenres)
        print(f"Letter {chr(i)} downloaded")
    dictGenres = dict(sorted(dictGenres.items(), key=lambda item: item[1], reverse=True))
    print(dictGenres)
    saveVariableInFile("./savedObjects/datasets/dictGenres", dictGenres)
    return dictGenres

def selectGenre(genres):
    genre = genres[0]
    if genre in choosedGenres:
        return genre
    else:
        return None

