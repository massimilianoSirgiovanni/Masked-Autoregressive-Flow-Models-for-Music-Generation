import random

from colorama import Fore, Style
import accuracyMetrics
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

    def trainModel(self, tr_set, val_set, batch_size, loss_function, parameter_funct=torch.nn.Module.parameters, num_epochs=100, patience=5, optimizer=optim.Adam, learning_rate=1e-3, file_path='./savedObjects/model', saveModel=True, beta=0.1):
        if batch_size > val_set.tensors[0].shape[0]:
            print(f"{Fore.LIGHTRED_EX}WARNING: The batch size is larger than the number of instances!{Style.RESET_ALL}")

        if self.tr_set == None:
            self.tr_set = data.DataLoader(tr_set, batch_size=batch_size, shuffle=True)

        if self.val_set == None:
            self.val_set = data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

        model = self.bestModel
        optimizer = optimizer(parameter_funct(model), lr=learning_rate)

        index_Patience = patience
        self.trainedEpochs = self.bestEpoch + 1
        for epoch in range(self.trainedEpochs, num_epochs):
            if len(self.trLossList) > self.trainedEpochs:
                self.trLossList = self.trLossList[0:self.trainedEpochs]
                self.valLossList = self.valLossList[0:self.trainedEpochs]
            tr_loss_epoch = []
            for (batch_data, batch_y) in tqdm(self.tr_set, desc='Training: '):
                batch_data = sparse_to_dense(batch_data).int()
                batch_y = sparse_to_dense(batch_y).int()
                with torch.autograd.detect_anomaly():
                    optimizer.zero_grad()
                    output = model(batch_data, batch_y)
                    tr_loss = loss_function(output, batch_data, beta=beta)
                    tr_loss.backward()
                    max_grad_norm = 1.0
                    torch.nn.utils.clip_grad_norm_(parameter_funct(model), max_grad_norm)
                    optimizer.step()
                    tr_loss_epoch.append(tr_loss.item())

            # Validation
            model.eval()
            with torch.no_grad():
                val_loss_epoch = []
                for (batch_data, batch_y) in tqdm(self.val_set, desc='Validation: '):
                    batch_data = sparse_to_dense(batch_data).int()
                    batch_y = sparse_to_dense(batch_y).int()
                    output_val = model(batch_data, batch_y)
                    val_loss = loss_function(output_val, batch_data, beta=beta)
                    val_loss_epoch.append(val_loss.item())

            self.trainedEpochs = epoch + 1
            print(np.array(tr_loss_epoch))
            print(np.array(val_loss_epoch))
            tr_loss_mean = tr_loss_epoch[-1]#numpy.array(tr_loss_epoch).mean()
            val_loss_mean = val_loss_epoch[-1]#numpy.array(val_loss_epoch).mean()
            print(tr_loss_mean)
            print(val_loss_mean)
            self.trLossList.append(tr_loss_mean)
            self.valLossList.append(val_loss_mean)
            if self.bestValLoss == None or self.bestValLoss >= val_loss_mean:
                self.bestValLoss = val_loss_mean
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

            print(f'Epoch [{epoch + 1}/{num_epochs}], Tr Loss: {tr_loss_mean:.4f}, Val Loss: {val_loss_mean:.4f}')
        if saveModel == True:
            saveVariableInFile(file_path, self)

    def generate(self, n_samples=1, u=None, genres=None):
        torch.seed()
        u = self.bestModel.generate(n_samples=n_samples, u=u, genres=genres)
        torch.manual_seed(seeds[0])
        return u

    def testModel(self, test_set, set_name="input", batch_size=100):
        test = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
        #inputX = test_set.tensors[0]
        #finalGeneration = None
        del test_set
        prec = 0
        acc = 0
        rec = 0
        for (batch_data, batch_y) in tqdm(test, desc=f'Test on {set_name}: '):
            batch_data = sparse_to_dense(batch_data).int()
            u, _ = self.bestModel(batch_data, batch_y); del _
            generatedX = self.generate(u=u, genres=batch_y); del u, batch_y
            '''if finalGeneration is None:
                finalGeneration = generatedX
            else:
                finalGeneration = torch.cat([finalGeneration, generatedX], dim=0)'''
            #finalGeneration.to_sparse_coo()
            prec += accuracyMetrics.precision_with_flatten(generatedX, batch_data)
            print(f"Precision: {accuracyMetrics.precision_with_flatten(generatedX, batch_data)}")
            rec += accuracyMetrics.recall_with_flatten(generatedX, batch_data)
            print(f"Recall: {accuracyMetrics.recall_with_flatten(generatedX, batch_data)}")
            acc += accuracyFunct(generatedX, batch_data)
            print(f"Similarity measure between {set_name} and midi generator: {accuracyFunct(generatedX, batch_data)}")
            del generatedX, batch_data
            print("End")
        print(len(test))
        print(f"Precision: {prec/len(test)}")
        print(f"Recall: {rec/len(test)}")
        accuracy = acc/len(test)
        print(f"Similarity measure between {set_name} and midi generator: {accuracy}")
        return accuracy

    '''def predict(self, test=None, seq_len=32):
        self.bestModel.eval()
        with torch.no_grad():
            if test == None:
                latent_sample = torch.randn(1, self.bestModel.encoder.latent_dim)
                prediction = self.bestModel.decode(latent_sample, seq_len)
                binary_output = binarize_predictions(torch.sigmoid(prediction), threshold=0.5)
            else:
                prediction = self.bestModel(test)
                binary_output = binarize_predictions(prediction[0], threshold=0.2)

        return binary_output'''


    def __str__(self):
        string = f"\nTRAINED MODEL:\n {str(self.bestModel)}"
        string = string + f"\nTRAINING STATS:\n-  Trained for: {self.trainedEpochs} Epochs\n-  Best Epoch: {self.bestEpoch}\n-  Best Validation Loss: {self.bestValLoss}\n"
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
            ymin = min(min(self.trLossList), min(self.valLossList))
            pp.vlines(x=self.bestEpoch, ymin=ymin, ymax=ymax, colors='tab:gray', linestyles='dashdot')
        pp.title(f"{type}Loss set through the epochs")
        pp.xlabel("Epochs")
        pp.ylabel(f"{type}Loss")

        temp_image_path = "./temp_plot.png"
        pp.savefig(temp_image_path)
        # Visualizza l'immagine
        display(Image(filename=temp_image_path))
        pp.draw()
        pp.show()




def holdoutSplit(X, Y, notes, val_percentage=0.1, test_percentage=0.1):
    if not exists(f'./savedObjects/datasets/{choosedInstrument}tr_dataset_program={choosedInstrument}') or not exists(
            f'./savedObjects/datasets/{choosedInstrument}val_dataset_program={choosedInstrument}') or not exists(
            f'./savedObjects/datasets/{choosedInstrument}test_dataset_program={choosedInstrument}'):
        X = sparse_to_dense(X)[:, :, notes]
        Y = sparse_to_dense(Y)
        N = X.shape[0]
        idx_rand = torch.randperm(N)
        N_val = int(N * val_percentage)
        N_te = int(N * test_percentage)
        N_tr = N - N_val - N_te
        idx_tr = idx_rand[:N_tr]
        idx_test = idx_rand[N_tr:N_tr + N_te]
        idx_val = idx_rand[N_tr + N_te:]
        X_tr, Y_tr = X[idx_tr], Y[idx_tr]
        X_val, Y_val = X[idx_val], Y[idx_val]
        X_test, Y_test = X[idx_test], Y[idx_test]
        tr_dataset = data.TensorDataset(X_tr.to_sparse_coo(), Y_tr.to_sparse_coo())
        saveVariableInFile(f'./savedObjects/datasets/tr_dataset_program={choosedInstrument}', tr_dataset)
        print(f"{Fore.CYAN}Training Set Size: {Fore.GREEN}{len(tr_dataset)}{Style.RESET_ALL}")
        val_dataset = data.TensorDataset(X_val.to_sparse_coo(), Y_val.to_sparse_coo())
        saveVariableInFile(f'./savedObjects/datasets/val_dataset_program={choosedInstrument}', val_dataset)
        print(f"{Fore.CYAN}Validation Set Size: {Fore.GREEN}{len(val_dataset)}{Style.RESET_ALL}")
        test_dataset = data.TensorDataset(X_test.to_sparse_coo(), Y_test.to_sparse_coo())
        saveVariableInFile(f'./savedObjects/datasets/test_dataset_program={choosedInstrument}', test_dataset)
        print(f"{Fore.CYAN}Test Set Size: {Fore.GREEN}{len(test_dataset)}{Style.RESET_ALL}")

    else:
        tr_dataset = loadVariableFromFile(f'./savedObjects/datasets/tr_dataset_program={choosedInstrument}')
        print(f"{Fore.CYAN}Training Set Size: {Fore.GREEN}{len(tr_dataset)}{Style.RESET_ALL}")
        val_dataset = loadVariableFromFile(f'./savedObjects/datasets/val_dataset_program={choosedInstrument}')
        print(f"{Fore.CYAN}Validation Set Size: {Fore.GREEN}{len(val_dataset)}{Style.RESET_ALL}")
        test_dataset = loadVariableFromFile(f'./savedObjects/datasets/test_dataset_program={choosedInstrument}')
        print(f"{Fore.CYAN}Test Set Size: {Fore.GREEN}{len(test_dataset)}{Style.RESET_ALL}")

    return tr_dataset, val_dataset, test_dataset

