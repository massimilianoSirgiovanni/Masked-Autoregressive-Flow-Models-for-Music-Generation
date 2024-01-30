from numpy.random import choice
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from config import choosedGenres
from genreAnalysis import convertGenreToNumber
from torch import manual_seed, logical_or

def plotPairwiseGenres(X, Y, first_genre, second_genre=None, max_samples=None, seed=None):
    first_genre = convertGenreToNumber(first_genre)
    if seed != None:
        manual_seed(seed)
    if second_genre != None:
        second_genre = convertGenreToNumber(second_genre)
        first_genre_mask = Y == first_genre
        second_genre_mask = Y == second_genre
        andMask = logical_or(first_genre_mask, second_genre_mask)
        Y = Y[andMask]
        X = X[andMask]
        second_genre_name = choosedGenres[second_genre]
    else:
        second_genre_name = 'others'
    first_genre_name = choosedGenres[first_genre]
    n_samples = X.shape[0]
    if max_samples != None or max_samples < n_samples:
        X, Y = chooseRandomElement(X, Y, max_samples)
        n_samples = max_samples
    X_reshaped = X.reshape(n_samples, -1)
    n_components = 2
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_reshaped)
    mask = Y == first_genre
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c='orange', marker='x', s=20, label=f'{first_genre_name}')
    plt.scatter(X_pca[~mask, 0], X_pca[~mask, 1], c='blue', marker='o', s=10, label=f'{second_genre_name}')
    plt.title('PCA Plot')
    plt.xlabel('PC 0')
    plt.ylabel('PC 1')
    plt.legend(title='Color Legend')
    plt.grid(True)
    plt.show()

def chooseRandomElement(tensor, conditon, n_sample):
    selected_indices = choice(tensor.shape[0], n_sample, replace=False)
    return tensor[selected_indices], conditon[selected_indices]

def noteFrequencyHinstogram(frequency_distribution):
    plt.bar(range(1, len(frequency_distribution) + 1), frequency_distribution, color='purple')
    plt.xlabel('Pitch Notes')
    plt.ylabel('Frequency')
    plt.title('Frequency distribution of notes in the piano roll')
    plt.show()

def plot_piano_roll(piano_roll, cmap='Blues', file_path=None):
    if len(piano_roll.shape) > 2:
        print(f"Warining! This function can plot only one piano roll at a time, so only the first one will be plotted")
        piano_roll = piano_roll[0]
    plt.imshow(piano_roll.numpy().T, aspect='auto', cmap=cmap, origin='lower', interpolation='none')
    plt.grid(True, linestyle='--', alpha=0.5, which='both', axis='both')
    plt.xlabel('Time (beat)')
    plt.ylabel('Pitch Notes')
    plt.title('Piano Roll')
    plt.colorbar(label='Note Pressure')
    if file_path != None:
        plt.savefig(file_path)
    plt.show()

def plot_loss(model, file_path=None):
    if len( model.valLossList) == 0 or len( model.trLossList) == 0:
        print(f"The Model must be trained first!")
        return -1
    plt.plot( model.trLossList, 'g-', label="training")
    plt.plot( model.valLossList, 'r--', label="validation")
    if  model.bestEpoch != None:
        ymax = max(max( model.trLossList), max(model.valLossList))
        ymin = min(min( model.trLossList), min(model.valLossList))
        plt.vlines(x= model.bestEpoch, ymin=ymin, ymax=ymax, colors='tab:gray', linestyles='dashdot')
    plt.title(f"{type}Loss set through the epochs")
    plt.xlabel("Epochs")
    plt.ylabel(f"{type}Loss")
    if file_path != None:
        plt.savefig(file_path)
    plt.draw()
    plt.show()

def plot_accuracy(model, file_path=None):
    plt.plot(model.accuracy)
    if model.bestEpoch != None:
        ymax = 1#max(model.accuracy)
        ymin = 0#min(model.accuracy)
        plt.vlines(x=model.bestEpoch, ymin=ymin, ymax=ymax, colors='tab:gray', linestyles='dashdot')
    plt.title(f"Training Accuracy set through the epochs")
    plt.xlabel("Epochs")
    plt.ylabel(f"Accuracy")

    if file_path != None:
        plt.savefig(file_path)
    plt.draw()
    plt.show()