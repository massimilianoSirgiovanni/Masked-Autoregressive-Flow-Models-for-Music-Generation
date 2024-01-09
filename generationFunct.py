import torch

from manageMIDI import *
from colorama import Fore, Style


def generateFromLatentSpace(model, latent_space, genre=torch.Tensor([0]), file_path="./output/song", instrument=0):
    print(genre)
    output = model.generate(n_samples=latent_space.shape[0], u=latent_space, genres=genre)
    print(output)
    end = latent_space.shape[0]
    if latent_space.shape[0] > 10:
        print(f"{Fore.YELLOW}WARNING: This method can only save 10 songs per time as a MIDI.\n      Therefore, only the first 10 will be saved{Style.RESET_ALL}")
        end = 10
    for i in range(0, end):
        dictionary = piano_roll_to_dictionary(output[i], instrument)
        newMidi = piano_roll_to_midi(dictionary)
        saveMIDI(newMidi, f"{file_path}Generated{i}.mid")
    return output


def latentSpaceInterpolation(model, first_song: tuple[torch.Tensor], second_song: tuple[torch.Tensor], interpolationFactor=0.5):
    u, _ = model(first_song[0], genres=first_song[1])
    z, _ = model(second_song[0], genres=second_song[1])
    return torch.lerp(z, u, interpolationFactor)



def generateAndSaveASong(model, song=None, genres=torch.Tensor([0]), file_path="./output/song", instrument=0):
    if song != None:
        genre = song[1]
        song = song[0]
        if song.shape[0]>1:
            print(f"{Fore.LIGHTMAGENTA_EX}This method can generate only a song for time, it will beh generated only the first one passed as an argument{Style.RESET_ALL}")
            song = song[0:1, :, :]
        dictionary = piano_roll_to_dictionary(song[0], instrument)
        newMidi = piano_roll_to_midi(dictionary)
        saveMIDI(newMidi, f"{file_path}Input.mid")
        print(genres)
        song_old, _ = model(song, genres=genre)
        song, _ = model(song, genres=genres)
        print(song_old)
        print(song)
    return generateFromLatentSpace(model, song, genres, file_path, instrument)

