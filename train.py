import random

from manageFiles import *
import torch.utils.data as data
from tqdm import tqdm
import torch
import torch.optim as optim
import matplotlib.pyplot as pp
from config import *
from manageMIDI import *
from IPython.display import display, Image

class trainingModel():
    def __init__(self, model):
        self.bestModel = model
        self.trainedEpochs = 0
        self.bestEpoch = -1
        self.valLossList = []
        self.trLossList = []
        self.bestValLoss = None
        self.tr_set = None
        self.val_set = None
        self.valReconList = []
        self.trReconList = []

    def trainModel(self, tr_set, val_set, te_set, batch_size, loss_function, parameter_funct=nn.Module.parameters, num_epochs=100, patience=5, optimizer=optim.Adam, learning_rate=1e-3, file_path='./savedObjects/model', saveModel=True, beta=0.1):
        if batch_size > tr_set.shape[0]:
            print(f"{Fore.LIGHTRED_EX}WARNING: The batch size is larger than the number of instances!{Style.RESET_ALL}")

        if self.tr_set == None:
            self.tr_set = data.DataLoader(tr_set, batch_size=batch_size, shuffle=True)

        if self.val_set == None:
            self.val_set = data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

        model = self.bestModel
        optimizer = optimizer(parameter_funct(model), lr=learning_rate)

        index_Patience = patience
        isRecon = False
        for epoch in range(self.bestEpoch + 1, num_epochs):
            for batch_data in tqdm(self.tr_set, desc='Training: '):
                with torch.autograd.detect_anomaly():
                    optimizer.zero_grad()
                    output = model(batch_data)
                    tr_loss = loss_function(output, batch_data, beta=beta)
                    if type(tr_loss) is tuple:
                        isRecon = True
                        tr_loss, tr_recon = tr_loss
                    tr_loss.backward()
                    max_grad_norm = 1.0
                    torch.nn.utils.clip_grad_norm_(parameter_funct(model), max_grad_norm)
                    optimizer.step()


            # Validation
            model.eval()
            with torch.no_grad():
                for batch_data in tqdm(self.val_set, desc='Validation: '):
                    output_val = model(batch_data)
                    val_loss = loss_function(output_val, batch_data, beta=beta)
                    if isRecon:
                        val_loss, val_recon = val_loss
            self.trainedEpochs = epoch + 1
            self.trLossList.append(tr_loss.item())
            self.valLossList.append(val_loss.item())
            if isRecon:
                self.trReconList.append(tr_recon.item())
                self.valReconList.append(val_recon.item())
            if self.bestValLoss == None or self.bestValLoss >= val_loss:
                self.bestValLoss = val_loss
                self.bestEpoch = epoch
                self.bestModel = model
                index_Patience = patience
                if saveModel == True:
                    saveVariableInFile(file_path, self)

            else:
                index_Patience -= 1
                if index_Patience <= 0:
                    print(f"-TRAINING STOPPED BY EARLY STOPPING PROCEDURE-")
                    break
                print(f"EARLY STOPPING: Patience Epochs remained {index_Patience}")

            out_string = f'Epoch [{epoch + 1}/{num_epochs}], Tr Loss: {tr_loss.item():.4f}, '
            if isRecon:
                out_string = out_string + f'Tr Recon: {tr_recon.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Recon: {val_recon.item():.4f}'
            else:
                out_string = out_string + f'Val Loss: {val_loss.item():.4f}'

            print(out_string)

        if saveModel == True:
            saveVariableInFile(file_path, self)

    def predict(self, test=None, seq_len=32):
        self.bestModel.eval()
        with torch.no_grad():
            if test == None:
                latent_sample = torch.randn(1, self.bestModel.encoder.latent_dim)
                prediction = self.bestModel.decode(latent_sample, seq_len)
                binary_output = binarize_predictions(torch.sigmoid(prediction), threshold=0.5)
            else:
                prediction = self.bestModel(test)
                binary_output = binarize_predictions(prediction[0], threshold=0.2)

        return binary_output


    def __str__(self):
        if len(self.valReconList) == 0:
            best_recon = None
        else:
            best_recon = self.valReconList[self.bestEpoch]
        string = f"\nTRAINED MODEL:\n {str(self.bestModel)}"
        string = string + f"\nTRAINING STATS:\n-  Trained for: {self.trainedEpochs} Epochs\n-  Best Epoch: {self.bestEpoch}\n-  Best Validation Loss: {self.bestValLoss}\n- Validation Reconstruction: {best_recon}\n"
        return string

    def plot_loss(self, type=''):
        if len(type) >= 2:
            type = type[0].upper() + type[1:] + " "
        if len(self.valLossList) == 0 or len(self.trLossList) == 0:
            print(f"The Model must be trained first!")
            return -1
        if type == 'validation':
            pp.plot(self.valLossList)
        elif type == 'training':
            pp.plot(self.trLossList)
        else:
            pp.plot(self.trLossList, 'g-', label="training")
            pp.plot(self.valLossList, 'r--', label="validation")
        if self.bestEpoch != None:
            ymax = max(max(self.trLossList), max(self.valLossList))
            pp.vlines(x=self.bestEpoch, ymin=0, ymax=ymax, colors='tab:gray', linestyles='dashdot')
        pp.title(f"{type}Loss set through the epochs")
        pp.xlabel("Epochs")
        pp.ylabel(f"{type}Loss")

        temp_image_path = "./temp_plot.png"
        pp.savefig(temp_image_path)
        # Visualizza l'immagine
        display(Image(filename=temp_image_path))
        pp.draw()
        pp.show()




def holdoutSplit(X, val_percentage=0.1, test_percentage=0.1):
    N = X.shape[0]
    idx_rand = torch.randperm(N)
    N_val = int(N * val_percentage)
    N_te = int(N * test_percentage)
    N_tr = N - N_val - N_te
    idx_tr = idx_rand[:N_tr]
    idx_test = idx_rand[N_tr:N_tr + N_te]
    idx_val = idx_rand[N_tr + N_te:]
    X_tr = X[idx_tr]
    X_val = X[idx_val]
    X_test = X[idx_test]

    return X_tr, X_val, X_test


def predictAndSaveASong(model, song):
    dictionary = piano_roll_to_dictionary(song[0], 0)
    newMidi = piano_roll_to_midi(dictionary)
    for i in newMidi.instruments:
        print(i)
        for note in i.notes:
            print(note)
    saveMIDI(newMidi, "./output/preVAE.mid")
    output = model.predict(song)
    print(output)
    dictionary = piano_roll_to_dictionary(output[0], 0)
    newMidi = piano_roll_to_midi(dictionary)
    saveMIDI(newMidi, "./output/vaeOutput.mid")
    for i in newMidi.instruments:
        print(i)
        for note in i.notes:
            print(note)
    print(newMidi)

def displaySong(model, te_set, j):
    print(f"Song: {j}")
    song = te_set[j:j + 1]
    predictAndSaveASong(model, song)