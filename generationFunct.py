from manageMIDI import *
from colorama import Fore, Style
from torch import Tensor, lerp


def generateFromLatentSpace(model, latent_space, genre=Tensor([0]), file_path="./output/song", instrument=0, seed=None):
    n_samples = latent_space.shape[0] if latent_space != None else 1
    output = model.generate(n_samples=n_samples, u=latent_space, genres=genre, seed=seed)
    end = n_samples
    if end > 10:
        print(f"{Fore.YELLOW}WARNING: This method can only save 10 songs per time as a MIDI.\n      Therefore, only the first 10 will be saved{Style.RESET_ALL}")
        end = 10
    for i in range(0, end):
        dictionary = piano_roll_to_dictionary(output[i], instrument)
        newMidi = piano_roll_to_midi(dictionary)
        saveMIDI(newMidi, f"{file_path}Generated{i}.mid")
    return output


def latentSpaceInterpolation(model, first_song: tuple[Tensor], second_song: tuple[Tensor], interpolationFactor=0.5):
    u, _ = model(first_song[0], genres=first_song[1])
    z, _ = model(second_song[0], genres=second_song[1])
    return lerp(z, u, interpolationFactor)



def generateAndSaveASong(model, song=None, genres=Tensor([0]), file_path="./output/song", instrument=0, seed=None):
    if song != None:
        genre = song[1]
        song = song[0]
        if song.shape[0]>1:
            print(f"{Fore.LIGHTMAGENTA_EX}This method can generate only a song for time, it will beh generated only the first one passed as an argument{Style.RESET_ALL}")
            song = song[0:1, :, :]
        dictionary = piano_roll_to_dictionary(song[0], instrument)
        newMidi = piano_roll_to_midi(dictionary)
        saveMIDI(newMidi, f"{file_path}Input.mid")
        song, _ = model(song, genres=genre)
        print(song)
    return generateFromLatentSpace(model, song, genres, file_path, instrument, seed=seed)

