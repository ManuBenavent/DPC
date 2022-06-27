from tsne_torch import TorchTSNE
import numpy as np
from matplotlib import pyplot
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
import random
import torch
import json

directory = 'ek_from_toyota'
dataset = 'ek'
colors = ['seagreen','royalblue','indianred','gold','teal','violet','lightgreen','slategray','pink','sienna']

def features_PCA(X):
    PCA_Mapping = PCA(n_components=2)
    PCA_Mapping = PCA_Mapping.fit(X)
    return PCA_Mapping.transform(X)

def features_TSNE(X):
    X = torch.Tensor(X)
    return TorchTSNE(n_iter=2500, n_components=2, perplexity=25.0, initial_dims=256, verbose=True).fit_transform(X)

def main(xfile: str, yfile: str, cdict, mode='tsne'):       
    X = np.loadtxt(xfile)
    labels = np.loadtxt(yfile)
    assert(len(X) == len(labels))

    idx = np.where(np.array(labels) < 8)
    labels = labels[idx]
    X = X[idx]

    if mode=='tsne':
        print('Using TSNE')
        Y = features_TSNE(X)
    elif mode=='pca':
        print('Using PCA')
        Y = features_PCA(X)
    else:
        raise ValueError('Mode must be tsne or pca.')

    fig, ax = pyplot.subplots()
    for g in np.unique(labels):
        ix = np.where(labels == g)
        ax.scatter(Y[ix, 0], Y[ix, 1], c = colors[int(g)], label = cdict[g], s = 20)

    lgd =ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pyplot.savefig(directory + '/' + mode + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')


if __name__ == '__main__':
    dic_ek = {0: 'rinse_board:cutting', 1: 'wipe_counter', 2: 'wash_pan', 3: 'dry_hand', 4: 'stir_food', 5: 'stir_vegetable', 6: 'stir_rice', 7: 'pour_water', 8: 'knead_dough', 9: 'chop_squash', 10: 'wipe_sink'}
    dic_toyota = {0: 'Cook.Usestove', 1: 'Makecoffee.Pourwater', 2: 'Getup', 3: 'Pour.Frombottle', 4: 'Eat.Snack', 5: 'Cook.Cut', 6: 'Makecoffee.Pourgrains', 7: 'Maketea.Boilwater', 8: 'WatchTV', 9: 'Pour.Fromkettle', 10: 'Usetelephone', 11: 'Leave', 12: 'Readbook', 13: 'Drink.Fromcan', 14: 'Cook.Cleandishes', 15: 'Drink.Fromglass', 16: 'Drink.Frombottle', 17: 'Enter', 18: 'Drink.Fromcup', 19: 'Cutbread', 20: 'Cook.Stir', 21: 'Laydown', 22: 'Uselaptop', 23: 'Walk', 24: 'Pour.Fromcan', 25: 'Maketea.Insertteabag', 26: 'Cook.Cleanup', 27: 'Eat.Attable', 28: 'Takepills', 29: 'Sitdown', 30: 'Usetablet'}

    
    xfile = directory + '/context.txt'
    yfile = directory + '/labels.txt'

    main(xfile, yfile, dic_toyota if dataset=='toyota' else dic_ek, mode='tsne')
    main(xfile, yfile, dic_toyota if dataset=='toyota' else dic_ek, mode='pca')