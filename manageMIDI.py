import torch as th
import mido
import os

def loadDataset(directory, max_time_steps, num_pitches):
    tensorList = []
    i = 0
    for filename in os.listdir(directory):
        midi_file = loadMIDIfromFile(directory, filename)
        tensorList.append(midi2tensor(midi_file, max_time_steps, num_pitches))
        i += 1
        if i == 6:
            break
    return th.stack(tensorList)

def loadMIDIfromFile(directory, fileName):
    file_path = f"./{directory}/{fileName}"
    return mido.MidiFile(file_path)

def midi2tensor(midi_file, max_time_steps, num_pitches):
    midi_tensor = th.zeros((max_time_steps, num_pitches))
    # Si analizza il file MIDI e viene riempito il tensore
    current_time_step = 0
    for msg in midi_file.play():
        if msg.type == 'note_on':
            pitch = msg.note
            velocity = msg.velocity
            midi_tensor[current_time_step, pitch] = velocity
            current_time_step += 1
        elif msg.type == 'note_off':
            # Si considera l'evento 'note_off' come la fine della nota
            pitch = msg.note
            midi_tensor[current_time_step, pitch] = 0
            current_time_step += 1

        # Uscita dal loop se è stato raggiunto il numero massimo di passi temporali
        if current_time_step >= max_time_steps:
            break

    return midi_tensor