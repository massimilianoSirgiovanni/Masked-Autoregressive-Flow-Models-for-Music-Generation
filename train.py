from colorama import Fore, Style
import accuracyMetrics
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from torch.optim import Adam
from config import *
from manageMIDI import *
from torch.nn import Module
from torch import no_grad, randperm, manual_seed
from manageMemory import cleanCache
from copy import deepcopy

class trainingModel(Module):
    def __init__(self, model):
        super().__init__()
        self.bestModel = model
        self.model = deepcopy(model)
        self.trainedEpochs = 0
        self.bestEpoch = -1
        self.valLossList = []
        self.trLossList = []
        self.bestValLoss = None

    def trainModel(self, tr_set, val_set, batch_size, parameter_funct=Module.parameters, num_epochs=100, patience=5, optimizer=Adam, learning_rate=1e-3, file_path=None, freeCache=True, clipping_value=0.1):
        if file_path == None:
            print(f"{Fore.LIGHTRED_EX}WARNING: The model will not be saved during training, to save it provide a \"file_path\" to the method{Style.RESET_ALL}")

        if batch_size > val_set.tensors[0].shape[0]:
            print(f"{Fore.LIGHTRED_EX}WARNING: The batch size is larger than the number of instances!{Style.RESET_ALL}")

        try:
            print(self.model)
        except:
            print("Set Early Stopping Attributes")
            self.model = deepcopy(self.bestModel)
            self.trainedEpochs = self.bestEpoch

        for i in range(0, len(self.trLossList)):
            try:
                self.trLossList[i] = self.trLossList[i].item()
                self.valLossList[i] = self.valLossList[i].item()
            except:
                pass

        tr_set = DataLoader(tr_set, batch_size=batch_size, shuffle=True)
        val_set = DataLoader(val_set, batch_size=batch_size, shuffle=True)
        optimizer = optimizer(parameter_funct(self.model), lr=learning_rate)

        for epoch in range(self.trainedEpochs, num_epochs):
            indexPatience = self.trainedEpochs - (self.bestEpoch + 1)
            if indexPatience >= patience:
                print(f"-TRAINING STOPPED BY EARLY STOPPING PROCEDURE-")
                break
            if len(self.trLossList) > self.trainedEpochs:
                self.trLossList = self.trLossList[0:self.trainedEpochs]
                self.valLossList = self.valLossList[0:self.trainedEpochs]
            self.model.train()
            for (batch_data, batch_y) in tqdm(tr_set, desc='Training: '):
                batch_data = batch_data.to(choosedDevice)
                batch_y = batch_y.to(choosedDevice)
                optimizer.zero_grad()
                _, tr_loss = self.model(batch_data, batch_y)
                cleanCache(freeCache)
                tr_loss.backward()
                clip_grad_norm_(self.model.parameters(), clipping_value)  # Gradient clipping
                optimizer.step()


            # Validation
            self.model.eval()
            with no_grad():
                for (batch_data, batch_y) in tqdm(val_set, desc='Validation: '):
                    cleanCache(freeCache)
                    batch_data = batch_data.to(choosedDevice)
                    batch_y = batch_y.to(choosedDevice)
                    _, val_loss = self.model(batch_data, batch_y)


            self.trainedEpochs = epoch + 1
            self.trLossList.append(tr_loss.item())
            self.valLossList.append(val_loss.item())
            if self.bestValLoss == None or self.bestValLoss >= val_loss:
                self.bestValLoss = val_loss
                self.bestEpoch = epoch
                self.bestModel = deepcopy(self.model)
                if file_path != None:
                    saveModel(self, file_path)

                #accuracyMetrics.completeAnalisysOnSongsSets(output, f"Epoch={epoch + 1}")

            else:
                if file_path != None:
                    saveModel(self, file_path)
                print(f"EARLY STOPPING: Patience Epochs remained {patience - (indexPatience + 1)}")

            print(f'Epoch [{epoch + 1}/{num_epochs}], Tr Loss: {tr_loss:.4f}, Val Loss: {val_loss:.4f}')
            # Debug
            #self.accuracy.append(self.testModel(tr_set, set_name="Training Set", batch_size=tr_set.tensors[0].shape[0]))

            ##################
        if file_path != None:
            saveModel(self, file_path)

    def generate(self, n_samples=1, u=None, genres=None, seed=None):
        with no_grad():
            x = self.bestModel.generate(n_samples=n_samples, u=u, genres=genres, seed=seed)
        return x

    def testModel(self, test_set, set_name="input", batch_size=100, freeCache=True):
        test = DataLoader(test_set, batch_size=batch_size, shuffle=True)
        n = len(test)
        prec = 0
        acc = 0
        rec = 0
        self.bestModel.eval()
        with no_grad():
            for (batch_data, batch_y) in tqdm(test, desc=f'Test on {set_name}: '):
                cleanCache(freeCache)
                batch_data = batch_data.to(choosedDevice)
                batch_y = batch_y.to(choosedDevice)
                u, _ = self.bestModel(batch_data, batch_y); del _
                generatedX = self.generate(u=u, genres=batch_y); del u, batch_y
                prec += accuracyMetrics.precision_with_flatten(generatedX, batch_data)
                rec += accuracyMetrics.recall_with_flatten(generatedX, batch_data)
                acc += accuracyFunct(generatedX, batch_data)
        accuracy = acc/n
        print(f"Similarity measures between {set_name} and midi generator: ")
        print(f"    > Precision: {Fore.LIGHTGREEN_EX}{prec / n}{Style.RESET_ALL}")
        print(f"    > Recall: {Fore.LIGHTGREEN_EX}{rec / n}{Style.RESET_ALL}")
        print(f"    > F1 Score: {Fore.LIGHTGREEN_EX}{accuracy}{Style.RESET_ALL}")

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




def holdoutSplit(X, Y, val_percentage=0.1, test_percentage=0.1, seed=None):
    if seed != None:
        manual_seed(seed)
    X = X.to_dense()
    N = X.shape[0]
    idx_rand = randperm(N)
    N_val = int(N * val_percentage)
    N_te = int(N * test_percentage)
    N_tr = N - N_val - N_te
    idx_tr = idx_rand[:N_tr]
    idx_test = idx_rand[N_tr:N_tr + N_te]
    idx_val = idx_rand[N_tr + N_te:]
    X_tr, Y_tr = X[idx_tr], Y[idx_tr]
    X_val, Y_val = X[idx_val], Y[idx_val]
    X_test, Y_test = X[idx_test], Y[idx_test]
    tr_dataset = TensorDataset(X_tr, Y_tr)
    print(f"{Fore.CYAN}Training Set Size: {Fore.GREEN}{len(tr_dataset)}{Style.RESET_ALL}")
    val_dataset = TensorDataset(X_val, Y_val)
    print(f"{Fore.CYAN}Validation Set Size: {Fore.GREEN}{len(val_dataset)}{Style.RESET_ALL}")
    test_dataset = TensorDataset(X_test, Y_test)
    print(f"{Fore.CYAN}Test Set Size: {Fore.GREEN}{len(test_dataset)}{Style.RESET_ALL}")

    return tr_dataset, val_dataset, test_dataset


def extractSong(set, number):
    return (set.tensors[0][number].to_dense().unsqueeze(0), set.tensors[1][number].to_dense())