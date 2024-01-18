import numpy
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from config import choosedGenres
from genreAnalysis import convertGenreToNumber
import pypianoroll

def plotPairwiseGenres(X, Y, first_genre, second_genre=None, max_samples=None, seed=None):
    first_genre = convertGenreToNumber(first_genre)
    if seed != None:
        torch.manual_seed(seed)
    if second_genre != None:
        second_genre = convertGenreToNumber(second_genre)
        first_genre_mask = Y == first_genre
        second_genre_mask = Y == second_genre
        andMask = torch.logical_or(first_genre_mask, second_genre_mask)
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
    selected_indices = np.random.choice(tensor.shape[0], n_sample, replace=False)
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