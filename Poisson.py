import pickle
import torch
import numpy as np
from torch import nn
import math
import matplotlib.pyplot as plt

import tensorflow as tf
import torch.nn.functional as F

def poissonLoss(pr, ob, kv):
    """Custom loss function for Poisson model."""
    "log Likeliness is to be minimized"

    p = pr
    o = torch.tensor([(float)(ob)],requires_grad=False).cuda()
    k = torch.tensor([(float)(kv)],requires_grad=False).cuda()


    a = torch.lgamma(torch.add(o,k))
    b = torch.neg(torch.lgamma(torch.add(o,1)))
    c = torch.neg(torch.lgamma(k))
    d = torch.mul(k,torch.log(k))
    e = torch.neg(torch.mul(k,torch.log(torch.add(k,p))))
    f = torch.mul(o,torch.log(p))
    g = torch.neg(torch.mul(o,torch.log(torch.add(k,o))))

    floss = torch.add(torch.add(torch.add(a,b),torch.add(c,d)),torch.add(torch.add(e,f),g))
    loss = torch.abs(floss)

    #loss = torch.abs(torch.sub(p,o))

    #print(loss)

    return loss

def mse(p, o):
    return abs(p-o)

class poisson(nn.Module):

    def __init__(self,size):
        super(poisson, self).__init__()

        self.coefficients = nn.Parameter(torch.Tensor(size).random_(to=1))
        self.bias = nn.Parameter(torch.Tensor(1).random_(to=1))

    def forward(self, input):

        return torch.exp(torch.add(self.bias,torch.dot(self.coefficients,input)))

class PossionRegression(nn.Module):

    def __init__(self):
        super().__init__()

        self.poisson = poisson(len(X[1]))

    def forward(self, x):

        output = self.poisson(x)

        return output

def train(net, epochs, lr, kval,print_every=6000):
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    criterion = poissonLoss

    inputs = torch.from_numpy(X).cuda()
    labels = torch.from_numpy(Y1).cuda()

    Cinputs = torch.from_numpy(CX).cuda()
    Clabels = torch.from_numpy(CY1).cuda()

    for e in range(epochs):

        val_losses = []
        val_mse = []

        for index in range(len(CX)):
            cinput = Cinputs[index]
            clabel = Clabels[index]

            output = net(cinput)

            loss = criterion(output, clabel,kval)
            #print(loss)
            val_losses.append(loss.item())
            val_mse.append(mse(output, clabel))

            #print(output.item(),clabel.item())

        print(np.mean(val_losses))
        print(math.sqrt(sum(val_mse)))
        # batch loop
        for index in range(len(X)):

            input = inputs[index]
            label = labels[index]

            net.zero_grad()
            # get the output from the model

            output = net(input)

            loss = criterion(output, label,kval)
            loss.backward()
            optimizer.step()

            #print(network.poisson.bias.grad)



    return np.mean(val_losses)







stuff = [10,10,10,10,10]
succes = []

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

pickle_in = open('Y1.pickle', 'rb')
CY1 = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('Y2.pickle', 'rb')
CY2 = pickle.load(pickle_in)
pickle_in.close()

for i in range(len(CY1)):
    if (CY1 == 0):
        CX.pop(i)
        CY1.pop(i)

for i in range(len(Y1)):
    if (Y1 == 0):
        X.pop(i)
        Y1.pop(i)


import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

num_bins = 50
n, bins, patches = plt.hist(Y1, num_bins, facecolor='blue', alpha=0.5)
plt.show()

extraX = X
CX = 1 * np.array(CX, dtype=np.float32)
CY1 = 1 * np.array(CY1, dtype=np.float32)
CY2 = 1 * np.array(CY2, dtype=np.float32)

X = 1 * np.array(X, dtype=np.float32)
Y1 = 1 * np.array(Y1, dtype=np.float32)
Y2 = 1 * np.array(Y2, dtype=np.float32)




network = PossionRegression()

network.cuda()

succes.append(train(network,20,0.001,kval = 0.1))

sum = 0
for bla in Y1:
    sum += bla
avg = sum/len(X)
sd = 0
for bla in Y1:
    sd += abs(avg - bla)
print(sd/len(X))


print(succes)