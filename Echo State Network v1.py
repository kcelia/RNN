import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random

import warnings
warnings.filterwarnings("ignore")


class Freq(Dataset):
    def __init__(self):
        pass
    
    def __len__(self):
        return 4096
        
    def __getitem__(self, i):
        freq = [100, 200, 300, 400, 500]
        y = np.random.choice(freq) #np.random.randint(0, 500)
        x = np.random.rand(501) * 3.
        x[y] = 100.
        start = np.random.randint(1, 299)
        return np.fft.ifft(x).real[start:start+200], freq.index(y)


class ESN(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, alpha =0.5, sparse_degree=0.5, spectral_radius=.95):
        super(ESN, self).__init__()
        self.alpha = alpha
        self.sparse_degree = sparse_degree
        self.w_in = torch.autograd.Variable(torch.randn((reservoir_size, input_size), requires_grad=False))
        self.w_hidden = torch.autograd.Variable(torch.randn(reservoir_size, reservoir_size), requires_grad=False)
        self.w_hidden *=(torch.rand((reservoir_size, reservoir_size)) <= sparse_degree).float()
        #normaliser par le rayon spectral
        #calcul des valeurs propres
        max_eigval = torch.eig(self.w_hidden, eigenvectors=False)[0].abs().max()
        #calcul de valeur spectral (max des valeurs propre)
        self.w_hidden = self.w_hidden *  spectral_radius / max_eigval
        self.w_out = torch.autograd.Variable(torch.randn((1 + reservoir_size + input_size, output_size)), requires_grad=True)
        
    def forward(self, x, hidden):
        # W : reservoir = hidden
        # new_hidden = (1 - alpha) * W(t) + alpha * tanh(W_in * x + W(t) * W(t-1)) 
        x_tild = nn.Tanh()(self.w_in @ x.t() + self.w_hidden @ hidden.t())
        new_hidden = (1 - self.alpha) * hidden + self.alpha * x_tild.t()
        #doute sur concatenation de x avec les autres
        y = torch.cat((torch.ones((x.shape[0], 1)), x, new_hidden), dim=1) @ self.w_out 
        y = nn.Softmax()(y)
        return y, new_hidden


# 
# vidéo : sequence multivariée
# premier test avec une séquence uni varié
# #u = x[:, 0, :]

batch_size = 64
input_size = 1 #courbe de taille longueur * 1
reservoir_size = 500
out_size = 5 #nb_classe
longueur = 200
hidden = torch.zeros((batch_size, reservoir_size))
x = torch.randn((batch_size, longueur, input_size))
loss_function = nn.CrossEntropyLoss()#nn.MSELoss() Transformer les y_target en OneHote 


train_loader = torch.utils.data.DataLoader(
    Freq(),
    shuffle=True,
    batch_size=batch_size
)


esn = ESN(input_size, reservoir_size, out_size, sparse_degree=.05, alpha=.2, spectral_radius=.9)
optimizer = torch.optim.Adam([esn.w_out])
from tqdm import tqdm

epochs = 50
train_history = []
for epochs in tqdm(range(epochs)):

    for x, y in train_loader:
        x = x.unsqueeze(2).float()

        hidden = torch.zeros((batch_size, reservoir_size)) #<!> sinon on prend l'ancien hidden
        optimizer.zero_grad()
        for l in range(x.shape[1]):
            y_hat, hidden = esn(x[:, l, :], hidden)

        loss = loss_function(y_hat, y)

        train_history.append(loss.item())
        loss.backward()
        optimizer.step()
    print((y_hat.argmax(1) == y).float().mean())
plt.plot(train_history)
plt.show()

accuracy = (y_hat.argmax(dim=1) == y).float().mean()
