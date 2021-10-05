
import pickle
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class Ordinal_loss(torch.nn.Module):
    #moddified binary cross entropy
    def __init__(self):
        super(Ordinal_loss, self).__init__()

    def forward(self, predicted, actual):
        BCE = torch.add(    torch.mul(actual,torch.log(predicted)) ,   torch.mul(  torch.sub(1,actual)   , torch.log(torch.sub(1,predicted))  ))
        return torch.neg(torch.sum(BCE))


class ANN(nn.Module):

    def __init__(self, full, out):
        super().__init__()

        self.fully1 = nn.Linear(full, 32)
        self.tany1 = nn.ReLU()
        self.fully2 = nn.Linear(32, out)
        self.tany2 = nn.Sigmoid()

    def forward(self, meta_data, lda_data):

        output = self.fully1(torch.cat((meta_data,lda_data)))
        output = self.tany1(output)
        output = self.fully2(output)
        output = self.tany2(output)

        return output


def train(net, epochs, lr, lda_data, clda_data, meta_data, cmeta_data, word_indexes, cword_indexes, Y, CY):
    # inputs ATLINEAR network, epochs, learning_rate, and the data split and in tensor form (.cuda)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    criterion = Ordinal_loss()

    epoch_loss = 99
    last_epoch_loss = 100

    for e in range(epochs):

        val_losses = []

        net.eval()

        for i in range(len(CY)):

            # print(cword_indexes[i])
            if(len(cword_indexes[i]) > 7):
                output = net(cmeta_data[i], clda_data[i])

                loss = criterion(output, CY[i])
                val_losses.append(loss.item())

        last_epoch_loss = epoch_loss
        epoch_loss = float(np.mean(val_losses))
        print(epoch_loss)

        if last_epoch_loss < epoch_loss:
            return

        net.train()

        for i in range(len(Y)):

            if (len(word_indexes[i]) > 7):
                net.zero_grad()
                output = net(meta_data[i], lda_data[i])

                # print(output)

                loss = criterion(output, Y[i])
                loss.backward()
                optimizer.step()


def main():
    pass

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    pickle_in = open('clda_data.pickle', 'rb')
    clda_data = pickle.load(pickle_in)
    pickle_in.close()
    pickle_in = open('lda_data.pickle', 'rb')
    lda_data = pickle.load(pickle_in)
    pickle_in.close()
    pickle_in = open('meta_data.pickle', 'rb')
    meta_data = pickle.load(pickle_in)
    pickle_in.close()
    pickle_in = open('cmeta_data.pickle', 'rb')
    cmeta_data = pickle.load(pickle_in)
    pickle_in.close()
    pickle_in = open('word_indexes.pickle', 'rb')
    word_indexes = pickle.load(pickle_in)
    pickle_in.close()
    pickle_in = open('cword_indexes.pickle', 'rb')
    cword_indexes = pickle.load(pickle_in)
    pickle_in.close()
    pickle_in = open('Y.pickle', 'rb')
    Y = pickle.load(pickle_in)
    pickle_in.close()
    pickle_in = open('CY.pickle', 'rb')
    CY = pickle.load(pickle_in)
    pickle_in.close()
    pickle_in = open('vocab_size.pickle', 'rb')
    vocab_size = pickle.load(pickle_in)
    pickle_in.close()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)

    # for i in cword_indexes:
    # i.to(device)
    # for i in word_indexes:
    # i.to(device)

    lda_data = lda_data.to(device)
    clda_data = clda_data.to(device)
    meta_data = meta_data.to(device)
    cmeta_data = cmeta_data.to(device)

    print(meta_data[0])

    Y = Y.to(device)
    CY = CY.to(device)

    print(Y.shape)

    ordinal_classes = Y.shape[1]
    full_feature_size = (lda_data.shape[1] + meta_data.shape[1])

    # vocab_size, embedding_dim, encoder_dim, attentioner_dim, full_feature_size, ordinal_classes
    ANN = ANN(full_feature_size, ordinal_classes)

    # net, epochs, lr, lda_data, clda_data, meta_data, cmeta_data, word_indexes, cword_indexes, Y, CY
    ANN = ANN.to(device)


    print("training....")


    #torch.save(ATLINEAR.state_dict(), "ATLINEAR_untrained.pickle")

    address = "ANN.pickle"
    print(address)
    torch.save(ANN.state_dict(), address)
    train(ANN, 50, 0.00001, lda_data, clda_data, meta_data, cmeta_data, word_indexes, cword_indexes, Y, CY)


    torch.save(ANN.state_dict(), address)

    #torch.save(ATLINEAR.state_dict(), "ATLINEAR_trained.pickle")
    main()


#[1,1,1,1]
#[0,0,0,0]
#[1,1,0,0]

