import torch
import pretty_midi
import os
from manageFiles import *
import random
from config import directory_dataset, current_directory, choosedGenres
from os import makedirs
from os.path import exists
from genreAnalysis import getDictGenre, getGenreFromId, gennreLabelToTensor
import pypianoroll
import numpy as np

default_instruments = range(0, 128)

def midiFilters(midi_data, time_signature_chosen = (4, 4)):
    # Get the time signature
    try:
        if len(midi_data.time_signature_changes) != 1:
            print(f"MIDI file change hid time signature during the execution")
        time_signature = (midi_data.time_signature_changes[0].numerator, midi_data.time_signature_changes[0].denominator)
    except:
        return False
    if time_signature != time_signature_chosen:
        print(f"MIDI file Tempo (={time_signature}) is different from the selected tempo (={time_signature_chosen})")
        return False
    return True



def midi_to_piano_roll(midi_data, ticks_per_beat=480, number_notes=128, num_bar=2, instruments=default_instruments):
    time_signature = (midi_data.time_signature_changes[0].numerator, midi_data.time_signature_changes[0].denominator)
    # Time synchronization of MIDI files
    midi_data.resolution = ticks_per_beat
    ticks_per_unit = int(ticks_per_beat/time_signature[0]) # Default case is a "semiquaver"
    # Initialize the piano roll's dimentions
    number_notes = number_notes + 3  # Added columns for: hold state, silent and ending
    time_steps = (num_bar * time_signature[0] * time_signature[1])
    piano_roll_dict = {}
    # Iterate through the notes and populate the piano roll
    for instrument in midi_data.instruments:
        program = instrument.program
        if program in instruments:
            start_time, end_time = 0, 0
            piano_roll = torch.zeros((1, time_steps, number_notes))
            for note in instrument.notes:
                # Calcolo numero Time Step
                start_time = int(midi_data.time_to_tick(note.start) / ticks_per_unit)
                end_time = int(midi_data.time_to_tick(note.end) / ticks_per_unit)
                if start_time >= time_steps:
                    break
                note_number = note.pitch
                piano_roll[0, start_time, note_number] = 1

                for i in range(start_time+1, end_time):
                    if i >= time_steps:
                        break
                    else:
                        piano_roll[0, i, number_notes-3] = 1
                        #print("Hold state")
            if end_time >= time_steps:
                end_track = time_steps-1
            else:
                end_track = end_time
            piano_roll[0, end_track, number_notes-1] = 1
            if torch.sum(piano_roll, (0, 1, 2)).item() != 1:
                for i in range(0, end_track):
                    if torch.sum(piano_roll[0, i], 0) == 0:
                        piano_roll[0, i, number_notes - 2] = 1
                if program in piano_roll_dict:
                    piano_roll_dict[program] = torch.cat((piano_roll_dict[program], piano_roll), dim=0)
                else:
                    piano_roll_dict[program] = piano_roll

    return piano_roll_dict

def removeConsecutiveSilence(pianoroll):
    consecutive_silence = torch.Tensor([128])
    consecutive_silence = consecutive_silence.repeat(1, 32, 1)
    index_consecutive = None
    for i in range(pianoroll.size(0) - consecutive_silence.size(0) + 1):
        if torch.equal(pianoroll[i:i + consecutive_silence.size(0)], consecutive_silence):
            index_consecutive = i
    if index_consecutive is not None:
        pianoroll = torch.cat([pianoroll[:index_consecutive,:,:], pianoroll[index_consecutive + 1:,:,:]], dim=0)
    return pianoroll

def piano_roll_reduced_representation(pianoroll):
    tensor_mask = (pianoroll != 0)
    pianoroll[~tensor_mask] = 2
    output = torch.argmin(pianoroll, dim=2).reshape(pianoroll.shape[0], pianoroll.shape[1], 1)
    #output = torch.argmax(pianoroll, dim=2).reshape(pianoroll.shape[0], pianoroll.shape[1], 1)
    return output

def getSlidingPianoRolls(pianoroll, num_bar=2, timestep_per_bar=16, binary=True):
    pianoroll = torch.Tensor(pianoroll)
    tail = pianoroll.shape[0] % timestep_per_bar
    pianoroll = pianoroll[:pianoroll.shape[0] - tail, :]
    pianoroll_unfolded = torch.nn.functional.unfold(pianoroll.unsqueeze(0), kernel_size=(num_bar, 1), dilation=(timestep_per_bar, 1), stride=1)
    pianoroll_unfolded = pianoroll_unfolded.reshape(num_bar, -1,  timestep_per_bar, 128)
    pianoroll = torch.stack([pianoroll_unfolded[i, :, :, :] for i in range(num_bar)], dim=1)
    pianoroll = pianoroll.reshape(pianoroll.shape[0], -1, pianoroll.shape[3])
    if binary:
        pianoroll[pianoroll != 0] = 1
    silent_state = torch.zeros((pianoroll.shape[0], pianoroll.shape[1], 1))
    pianoroll = torch.cat([pianoroll, silent_state], dim=2)
    pianoroll[:, :, -1] = torch.sum(pianoroll, dim=2) == 0
    return pianoroll

def loadDataset(directory):
    print(f"Loading Dataset from {directory} ...")
    dataset_midi = {}
    dictGenre = getDictGenre("./amg")
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(".npz")]:
            try:
                genre = getGenreFromId(dirpath[-18:], dictGenre=dictGenre)
                if genre is None:
                    raise Exception("Song not has genre")
                else:
                    midi_path = f"{dirpath}/{filename}"
                    midi_path = midi_path.replace("lpd/lpd_cleansed", "lmd_matched").replace("npz", "mid")
                    midi_file = pretty_midi.PrettyMIDI(midi_path)
                    if midiFilters(midi_file):
                        multitrack = pypianoroll.from_pretty_midi(midi_file, resolution=4)
                        for track in multitrack.tracks:
                            piano_rolls = getSlidingPianoRolls(track.pianoroll)
                            piano_rolls = piano_roll_reduced_representation(piano_rolls)
                            piano_rolls = torch.unique(piano_rolls, dim=0) # Remove Duplicates
                            piano_rolls = removeConsecutiveSilence(piano_rolls)
                            program = track.program
                            if program in dataset_midi:
                                dataset_midi[program][0] = torch.cat((dataset_midi[program][0], piano_rolls))
                            else:
                                dataset_midi[program] = [piano_rolls, []]
                            for j in range(0, piano_rolls.shape[0]):
                                dataset_midi[program][1].append(genre)
            except Exception as e:
                print(f"Skipped {dirpath}/{filename} cause of:\n{e}")
                pass
    return dataset_midi


def loadLMD(directory, starting_point="A", ending_point="Z", instruments=default_instruments):
    for i in range(ord(starting_point), ord(ending_point) + 1):
        checkFile = True
        if not exists(f"{directory_dataset}/{chr(i)}"):
            makedirs(f"{directory_dataset}/{chr(i)}")
            checkFile = False
        if checkFile == False or databaseLoadedInstruments(f"{directory_dataset}/{chr(i)}", instruments) == False:
            tmp = loadDataset(f"{current_directory}/{directory}/{chr(i)}")
            for instr in instruments:
                if instr in tmp:
                    saveVariableInFile(f"{directory_dataset}/{chr(i)}/dataset_program={instr}", tmp[instr])
                    print(tmp[instr][0].shape)
                else:
                    saveVariableInFile(f"{directory_dataset}/{chr(i)}/dataset_program={instr}", None)
                    print(None)
        print(f"Letter {chr(i)} downloaded")
    return True

def databaseLoadedInstruments(directory, instruments):
    for i in instruments:
        if os.path.isfile(f"{directory}/dataset_program={i}") == False:
            return False
    return True

def transformListMidi(listMidi, program=default_instruments):
    listTensor = []
    for midi_file in listMidi:
        try:
            listTensor.append(midi_to_piano_roll(midi_file, instruments=program))
        except Exception as e:
            print(f"Skipped cause of:\n{e}")
            pass
    return torch.stack(listTensor)


def piano_roll_to_dictionary(piano_roll, program, dictionary=None):
    if dictionary == None:
        dictionary = {}
    if type(piano_roll) != list:
        piano_roll = [piano_roll]

    dictionary[program] = piano_roll

    return dictionary

def piano_roll_to_midi(piano_roll_dictionary, tick_per_beat=480, time_signature=(4, 4)):
    # piano_roll_dictionary it is a dictionary that contains the instruments as keys and a list of piano rolls as values
    midi_data = pretty_midi.PrettyMIDI()
    midi_data.resolution = tick_per_beat
    midi_data.time_signature_changes = [
    pretty_midi.TimeSignature(numerator=time_signature[0], denominator=time_signature[1], time=0)]
    #tick_per_unit = tick_per_beat / time_signature[1]
    for i in piano_roll_dictionary:
        for piano_roll in piano_roll_dictionary[i]:
            try:
                instrument = piano_roll_to_instrument(piano_roll, midi_data, program=i)
                midi_data.instruments.append(instrument)
            except Exception as e:
                print(f"Instrument {i} wasn't loaded couse of:\n{e}")
    return midi_data

def piano_roll_to_instrument(piano_roll, midi_data,  program=default_instruments[0]):
    tick_per_unit = midi_data.resolution / midi_data.time_signature_changes[0].denominator
    instrument = pretty_midi.Instrument(program=program)
    hold = 0
    for i in range(piano_roll.shape[0], 0, -1):
        end_time = midi_data.tick_to_time(int(i*tick_per_unit)) + midi_data.tick_to_time(int(hold*tick_per_unit))
        notes = (piano_roll[i-1, :] == 1).nonzero(as_tuple=True)[0]
        if piano_roll.shape[1]-3 not in notes:
            hold = 0
        else:
            hold += 1
            notes = notes[notes!=piano_roll.shape[1]-3]
        for j in range(len(notes)-1, -1, -1):
            pitch = int(notes[j])
            velocity = random.randint(40, 100)
            if pitch == piano_roll.shape[1]-2:
                pass
            elif pitch == piano_roll.shape[1]-1:
                break
            else:
                start_time = midi_data.tick_to_time(int((i-1)*tick_per_unit))
                note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time)
                instrument.notes.insert(0, note)
    return instrument


def removeDuplicatesNoConsideringGenres(dataset, genres):
    index = np.unique(dataset, return_index=True, axis=0)[1]

    return dataset[index], genres[index]


def getSingleInstrumentDatabaseLMD(directory, instrument):
    dataset = None
    genre = None
    for i in range(ord("A"), ord("Z") + 1):
        if os.path.isfile(f"{directory}/{chr(i)}/dataset_program={instrument}"):
            input = loadVariableFromFile(f"{directory}/{chr(i)}/dataset_program={instrument}")
            if input != None:
                tmp, genreTmp = input
                genreTmp = gennreLabelToTensor(genreTmp, choosedGenres)
                genreTmp = genreTmp.unsqueeze(1).unsqueeze(2).expand(-1, tmp.shape[1], 1)
                tmp = torch.cat([tmp, genreTmp], dim=2)
                #tmp, genreTmp = removeDuplicates(tmp, genreTmp)
                tmp = torch.unique(tmp, dim=0)  # Remove Duplicatesù
                if dataset != None:
                    dataset = torch.cat((dataset, tmp))
                    #genre = torch.cat((genre, genreTmp))
                    #dataset, genre = removeDuplicates(dataset, genre)
                    dataset = torch.unique(dataset, dim=0)  # Remove Duplicates
                    print(f"Complete Dataset: {dataset.shape}")
                else:
                    dataset = tmp
                    genre = genreTmp

    if dataset != None:
        newDataset = torch.unique(dataset, dim=0)  # Remove Duplicates
        dataset = newDataset[:, :, :-1]
        genre = newDataset[:, 0, -1]
        saveVariableInFile(f"{directory}/dataset_complete_program={instrument}", (dataset, genre))
    else:
        print(f"There are no tracks associated with the instrument={instrument}")
    return dataset, genre

def binarize_predictions(predictions, threshold=0.5):
    # Applica la soglia di attivazione
    binary_predictions = (predictions > threshold).int()
    return binary_predictions

