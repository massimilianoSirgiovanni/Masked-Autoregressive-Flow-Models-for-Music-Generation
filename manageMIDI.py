import torch
import pretty_midi
import os
from manageFiles import *
import random

default_instruments = [5]

def midiFilters(midi_data, time_signature_chosen = (4, 4), instruments=default_instruments):
    # Get the time signature
    try:
        time_signature = (midi_data.time_signature_changes[0].numerator, midi_data.time_signature_changes[0].denominator)
    except:
        return False
    if time_signature != time_signature_chosen:
        print(f"MIDI file Tempo (={time_signature}) is different from the selected tempo (={time_signature_chosen})")
        return False
    '''output = False
    for i in midi_data.instruments:
        if i.program in instruments:
            output = True
            break'''
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
    piano_roll_array = []
    # Iterate through the notes and populate the piano roll
    for instrument in midi_data.instruments:
        if instrument.program in instruments:
            piano_roll = torch.zeros((time_steps, number_notes))
            start_time, end_time = 0, 0
            for note in instrument.notes:
                # Calcolo numero Time Step
                start_time = int(midi_data.time_to_tick(note.start) / ticks_per_unit)
                if start_time > end_time:
                    for i in range(end_time, start_time):
                        if i >= 32:
                            break
                        else:
                            piano_roll[i, number_notes - 2] = 1
                            #print("Silent state")
                end_time = int(midi_data.time_to_tick(note.end) / ticks_per_unit)
                if start_time >= time_steps:
                    break
                note_number = note.pitch
                piano_roll[start_time, note_number] = 1

                for i in range(start_time+1, end_time):
                    if i >= time_steps:
                        break
                    else:
                        piano_roll[i, number_notes-3] = 1
                        #print("Hold state")
            if end_time >= time_steps:
                piano_roll[time_steps-1, number_notes-1] = 1
            else:
                piano_roll[end_time, number_notes-1] = 1
            piano_roll_array.append(piano_roll)

    return piano_roll_array

def loadDataset(directory):
    print(f"Loading Dataset from {directory} ...")
    dataset_midi = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(".mid")]:
            print(f"{dirpath}/{filename}")
            try:
                midi_file = pretty_midi.PrettyMIDI(f"{dirpath}/{filename}")
                if midiFilters(midi_file):
                    #print(f"Inserted: {filename}")
                    dataset_midi.append(midi_file)
            except Exception as e:
                print(f"Skipped {dirpath}/{filename} cause of:\n{e}")
                pass
    return dataset_midi

def loadLMD(directory, starting_point="A", ending_point="Z"):
    if os.path.isfile("./variables/dataset"):
        dataset = loadVariableFromFile("./variables/dataset")
    else:
        dataset = loadDataset(f"./{directory}/{starting_point}")
        saveVariableInFile("./variables/dataset", dataset)
        print(f"Letter {starting_point} downloaded")
        starting_point = chr(ord(starting_point) + 1)
    for i in range(ord(starting_point), ord(ending_point) + 1):
        tmp = loadDataset(f"./{directory}/{chr(i)}")
        if torch.is_tensor(tmp):
            dataset = torch.cat(dataset, tmp)
            saveVariableInFile("./variables/dataset", dataset)
        print(f"Letter {chr(i)} downloaded")

def transformListMidi(listMidi, program=default_instruments):
    listTensor = []
    for midi_file in listMidi:
        try:
            listTensor.append(midi_to_piano_roll(midi_file, instruments=program))
        except Exception as e:
            print(f"Skipped cause of:\n{e}")
            pass
    return torch.stack(listTensor)


def piano_roll_to_dictionary(dictionary, piano_roll, program):
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
                instrument = piano_roll_to_instrument(piano_roll, midi_data, program=[i])
                midi_data.instruments.append(instrument)
            except Exception as e:
                print(f"Instrument {i} wasn't loaded couse of:\n{e}")
    return midi_data

def piano_roll_to_instrument(piano_roll, midi_data,  program=default_instruments):
    # Create a PrettyMIDI object
    tick_per_unit = midi_data.resolution / midi_data.time_signature_changes[0].denominator
    for i in program:
        instrument = pretty_midi.Instrument(program=i)
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





