import pickle
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

import tensorflow as tf
import torch.nn.functional as F

class LSTMeaning(nn.Module):

    def __init__(self):
        super().__init__()

        self.n_layers = 1
        self.batch_size = 1
        self.seq_length = 30
        self.n_hidden = 5
        self.input_size = 300

        self.LSTM = nn.LSTM(self.input_size,self.n_hidden)

        self.fully1 = nn.Linear(len(X[1])+self.n_hidden, 32)

        self.tany = nn.Sigmoid()

        self.fully2 = nn.Linear(32, 1)


    def forward(self, veckos, x, hidden):

        print(veckos.shape)

        wacko, hidden = self.LSTM(veckos.view(30, 1, 300), hidden)

        print(wacko.shape)

        comb = torch.cat((wacko.contiguous().view(),x))

        output = self.fully1(comb)

        output = self.tany(output)

        output = self.fully2(output)

        return output

    def init_hidden(self):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data


        hidden = (weight.new(self.n_layers, self.batch_size, self.n_hidden).zero_(),
                  weight.new(self.n_layers, self.batch_size, self.n_hidden).zero_())

        return hidden

def train(net, epochs, lr, print_every=6000):
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    criterion = nn.L1Loss()

    inputs = torch.from_numpy(X)
    labels = torch.from_numpy(Y1)
    Vecs_inputs = torch.from_numpy(Vecs)

    Cinputs = torch.from_numpy(CX)
    Clabels = torch.from_numpy(CY1)
    CVecs_inputs = torch.from_numpy(CVecs)

    for e in range(epochs):

        h = net.init_hidden()

        val_losses = []

        for index in range(len(CX)):
            h = tuple([each.data for each in h])

            cinput = Cinputs[index]
            clabel = Clabels[index]
            CVecs_input = CVecs_inputs[index]

            output, h = net(CVecs_input,cinput,h)

            loss = criterion(output, clabel)
            val_losses.append(loss.item())
            nn.utils.clip_grad_norm_(net.parameters(), 5)

        print(np.mean(val_losses))

        Ch= net.init_hidden()
        # batch loop
        for index in range(len(X)):
            Ch = tuple([each.data for each in Ch])

            input = inputs[index]
            label = labels[index]
            Vecs_input = Vecs_inputs[index]

            net.zero_grad()
            # get the output from the model

            output, Ch = net(Vecs_input, input,Ch)
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

pickle_in = open('clean_text.pickle', 'rb')
clean_text = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('word_vectors.pickle', 'rb')
word_vectors = pickle.load(pickle_in)
pickle_in.close()


Vecs = word_vectors[:3000]
CVecs = word_vectors[3000:]

extraX = X
CX = 1 * np.array(CX, dtype=np.float32)
CY1 = 1 * np.array(CY1, dtype=np.float32)
CY2 = 1 * np.array(CY2, dtype=np.float32)
CVecs = 1 * np.array(CVecs, dtype=np.float32)

X = 1 * np.array(X, dtype=np.float32)
Y1 = 1 * np.array(Y1, dtype=np.float32)
Y2 = 1 * np.array(Y2, dtype=np.float32)
Vecs = 1 * np.array(Vecs, dtype=np.float32)

network = LSTMeaning()

network

succes.append(train(network,30,0.001))

sum = 0
for bla in Y1:
    sum += bla
avg = sum/len(X)
sd = 0
for bla in Y1:
    sd += abs(avg - bla)
print(sd/len(X))



print(succes)