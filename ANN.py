import pickle
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

import tensorflow as tf
import torch.nn.functional as F

class ANN(nn.Module):

    def __init__(self):
        super().__init__()

        self.fully1 = nn.Linear(len(X[1]), 32)
        self.tany = nn.Sigmoid()
        self.fully2 = nn.Linear(32, 1)

    def forward(self, x):

        output = torch.exp(self.fully2(self.tany(self.fully1(x))))

        return output

def train(net, epochs, lr, print_every=6000):
    net.train()

    epoch_loss = 99
    last_epoch_loss = 100

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    criterion = nn.L1Loss()
    #criterion = nn.MSELoss()

    inputs = torch.from_numpy(X).cuda()
    labels = torch.from_numpy(Y1).cuda()

    Cinputs = torch.from_numpy(CX).cuda()
    Clabels = torch.from_numpy(CY1).cuda()

    for e in range(epochs):

        val_losses = []

        for index in range(len(CX)):
            cinput = Cinputs[index]
            clabel = Clabels[index]

            output = net(cinput)

            loss = criterion(output, clabel)
            val_losses.append(loss.item())

            print(clabel, output)

        print(np.mean(val_losses))

        if last_epoch_loss < epoch_loss:
            return

        # batch loop
        for index in range(len(X)):

            input = inputs[index]
            label = labels[index]

            net.zero_grad()
            # get the output from the model

            output = net(input)
            #print(output)
            #print(output)
            # calculate the loss and perform backprop
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

    return np.mean(val_losses)

stuff = [10,10,10,10,10]
succes = [10]

pickle_in = open('X.pickle', 'rb')
X = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('LDA.pickle', 'rb')
LDA = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('Y1.pickle', 'rb')
Y1 = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('Y2.pickle', 'rb')
Y2 = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('CX.pickle', 'rb')
CX = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('CY1.pickle', 'rb')
CY1 = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('CY2.pickle', 'rb')
CY2 = pickle.load(pickle_in)
pickle_in.close()

print(CX)
extraX = X
CX = 1 * np.array(CX, dtype=np.float32)
CY1 = 1 * np.array(CY1, dtype=np.float32)
CY2 = 1 * np.array(CY2, dtype=np.float32)

X = 1 * np.array(X, dtype=np.float32)
Y1 = 1 * np.array(Y1, dtype=np.float32)
Y2 = 1 * np.array(Y2, dtype=np.float32)

network = ANN()

network.cuda()

train(network,30,0.0001)

address = "ANN.pickle"
print(address)

torch.save(network.state_dict(), address)




print(succes)