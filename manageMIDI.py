import torch
import pretty_midi
import os
from manageFiles import *
import random
from config import directory_dataset, current_directory
from os import makedirs
from os.path import exists

default_instruments = range(0, 128)

def midiFilters(midi_data, time_signature_chosen = (4, 4)):
    # Get the time signature
    try:
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
    #print(f"ticks_per_beat={ticks_per_beat}")
    ticks_per_unit = int(ticks_per_beat/time_signature[0]) # Default case is a "semiquaver"
    #print(f"ticks_per_unit={ticks_per_unit}")
    # Initialize the piano roll's dimentions
    number_notes = number_notes + 3  # Added columns for: hold state, silent and ending
    time_steps = (num_bar * time_signature[0] * time_signature[1])
    piano_roll_dict = {}
    # Iterate through the notes and populate the piano roll
    for instrument in midi_data.instruments:
        program = instrument.program
        if program in instruments:
            piano_roll = torch.zeros((1, time_steps, number_notes))
            start_time, end_time = 0, 0
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
                        #print("Silent State")
                if program in piano_roll_dict:
                    piano_roll_dict[program] = torch.cat((piano_roll_dict[program], piano_roll))
                else:
                    piano_roll_dict[program] = piano_roll

    return piano_roll_dict

def loadDataset(directory, instruments=default_instruments):
    print(f"Loading Dataset from {directory} ...")
    dataset_midi = {}
    p = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(".mid")]:
            #print(f"{dirpath}/{filename}")
            try:
                midi_file = pretty_midi.PrettyMIDI(f"{dirpath}/{filename}")
                if midiFilters(midi_file):
                    #print(f"Inserted: {filename}")
                    #dataset_midi.append(f"{dirpath}/{filename}")
                    local_dict = midi_to_piano_roll(midi_file, instruments=instruments)
                    for i in instruments:
                        if i in local_dict:
                            if i in dataset_midi:
                                dataset_midi[i] = torch.cat((dataset_midi[i], local_dict[i]))
                            else:
                                dataset_midi[i] = local_dict[i]

            except Exception as e:
                print(f"Skipped {dirpath}/{filename} cause of:\n{e}")
                pass

    return dataset_midi
    #torch.stack(dataset_midi, dim=0)

def loadLMD(directory, starting_point="A", ending_point="Z", instruments=default_instruments):
    for i in range(ord(starting_point), ord(ending_point) + 1):
        checkFile = True
        if not exists(f"{directory_dataset}/{chr(i)}"):
            makedirs(f"{directory_dataset}/{chr(i)}")
            checkFile = False
        if checkFile == False or databaseLoadedInstruments(f"{directory_dataset}/{chr(i)}", instruments) == False:
            tmp = loadDataset(f"{current_directory}/{directory}/{chr(i)}", instruments=instruments)
            for instr in instruments:
                if instr in tmp:
                    saveVariableInFile(f"{directory_dataset}/{chr(i)}/dataset_program={instr}", tmp[instr])
                else:
                    saveVariableInFile(f"{directory_dataset}/{chr(i)}/dataset_program={instr}", None)
        print(f"Letter {chr(i)} downloaded")
    return True

def databaseLoadedInstruments(directory, instruments):
    for i in instruments:
        if os.path.isfile(f"{directory}/dataset_program={i}") == False:
            return False
    return True


'''def loadLMD(directory, starting_point="A", ending_point="Z"):
    dataset = None
    if os.path.isfile(f"{current_directory}/savedObjects/datasets/dataset"):
        return loadVariableFromFile(f"{current_directory}/savedObjects/datasets/dataset")
    for i in range(ord(starting_point), ord(ending_point) + 1):
        if os.path.isfile(f"{directory_dataset}/{chr(i)}"):
            tmp = loadVariableFromFile(f"{directory_dataset}/{chr(i)}")
        else:
            tmp = loadDataset(f"{current_directory}/{directory}/{chr(i)}")
            saveVariableInFile(f"{directory_dataset}/{chr(i)}", tmp)
        if dataset == None:
            dataset = tmp
        else:
            if torch.is_tensor(tmp):
                dataset = torch.cat((dataset, tmp))
        print(f"Letter {chr(i)} downloaded")
    saveVariableInFile(f"{current_directory}/savedObjects/datasets/dataset", dataset)

    return dataset'''

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



def getSingleInstrumentDatabaseLMD(directory, instrument):
    dataset = None
    for i in range(ord("A"), ord("Z") + 1):
        if os.path.isfile(f"{directory}/{chr(i)}/dataset_program={instrument}"):
            tmp = loadVariableFromFile(f"{directory}/{chr(i)}/dataset_program={instrument}")
            if tmp != None:
                print(tmp.shape)
                if dataset != None:
                    dataset = torch.cat((dataset, tmp))
                else:
                    dataset = tmp
    if dataset != None:
        saveVariableInFile(f"{directory}/dataset_complete_program={instrument}", dataset)
    else:
        print(f"There are no tracks associated with the instrument={instrument}")
    return dataset

def binarize_predictions(predictions, threshold=0.5):
    # Applica la soglia di attivazione
    binary_predictions = (predictions > threshold).int()
    return binary_predictions

