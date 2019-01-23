import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
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

class CelluleRnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CelluleRnn, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.predict = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )
        
    def forward(self, x, hidden):
        n = torch.cat((x.double(), hidden.double()), dim=1).float()
        hidden = self.seq(n)
        y = self.predict(hidden)
        return y, hidden

class MyRnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer):
        super(MyRnn, self).__init__()
        self.num_layer = num_layer
        self.cells = 
            [CelluleRnn(input_size, hidden_size, hidden_size)] + 
            [
            CelluleRnn(hidden_size, hidden_size, output_size) 
            for i in range(self.num_layer - 1)
            ]

    def forward(self, x, hidden):
            new_hidden = x
            lst = []
            for i in range(len(self.cells)):
                y, new_hidden = self.cells[i](new_hidden, hidden[i])
                lst.append(new_hidden)
            return y, lst


batch_size = 64
input_size = 1 #courbe de taille longueur * 1
reservoir_size = 500
out_size = 5   #nb_classe
longueur = 200
hidden_size = 4
epochs = 50

loss_function = nn.CrossEntropyLoss()

train_loader = torch.utils.data.DataLoader(
    Freq(),
    shuffle=True,
    batch_size=batch_size
)

model1 = CelluleRnn(input_size, hidden_size=4, output_size=5)
optimizer = torch.optim.Adam(model1.parameters())

model = MyRnn(input_size, hidden_size=4, output_size=5, num_layer=4)

num_layer = 4
p = [v for c in model.cells for v in c.parameters()]
optimizer = torch.optim.Adam(p)

train_history = []

for x, y in train_loader:
    x = x.unsqueeze(2).float()
    optimizer.zero_grad()
    hidden = [torch.zeros((batch_size,hidden_size)) for _ in range(num_layer)]
    for l in range(x.shape[1]):
        y_hat, hidden = model(x[:, l, :], hidden)
    loss = loss_function(y_hat, y)

    train_history.append(loss.item())
    loss.backward()
    optimizer.step()

plt.plot(train_history)
plt.show()

accuracy = (y_hat.argmax(dim=1) == y).float().mean()
