import pickle
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel



class ANNer(nn.Module):
    def __init__(self, feature_size, ordinal_classes):
        super().__init__()
        self.feature_size = feature_size
        self.ordinal_classes = ordinal_classes

        self.linear = nn.Linear(self.feature_size, self.ordinal_classes)
        self.sigy = nn.Sigmoid()

    def forward(self, inputs):
        # inputs tensor of shape (metadata + LDA + specific_words + chanel num, 1)
        output = self.linear(inputs)
        output = self.sigy(output)
        # outputs tensor of shape (ordinal_classes, 1)
        return output

class Ordinal_loss(torch.nn.Module):
    #moddified binary cross entropy
    def __init__(self):
        super(Ordinal_loss, self).__init__()

    def forward(self, predicted, actual):
        BCE = torch.add(    torch.mul(actual,torch.log(predicted)) ,   torch.mul(  torch.sub(1,actual)   , torch.log(torch.sub(1,predicted))  ))
        return torch.neg(torch.sum(BCE))

class TRANSFORM(nn.Module):
    def __init__(self, feature_size, ordinal_classes):
        super().__init__()

        model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, inputs):
        # inputs tensor of shape (metadata + LDA + specific_words + chanel num, 1)

        # outputs tensor of shape (ordinal_classes, 1)
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

            output = net(cword_indexes[i], cmeta_data[i], clda_data[i], CY[i])

            loss = criterion(output, CY[i])
            val_losses.append(loss.item())

        last_epoch_loss = epoch_loss
        epoch_loss = float(np.mean(val_losses))
        print(epoch_loss)

        if last_epoch_loss < epoch_loss:
            return

        net.train()

        for i in range(len(Y)):

            net.zero_grad()
            output = net(word_indexes[i], meta_data[i], lda_data[i], Y[i])

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

    ngram_s = 7
    ordinal_classes = Y.shape[1]
    full_feature_size = (lda_data.shape[1] + meta_data.shape[1])
    embedding_dim = 256
    cnn_channels = 50

    # vocab_size, embedding_dim, encoder_dim, attentioner_dim, full_feature_size, ordinal_classes
    CNN = CNN(vocab_size + 1, embedding_dim, full_feature_size, ordinal_classes, cnn_channels, ngram_s)

    # net, epochs, lr, lda_data, clda_data, meta_data, cmeta_data, word_indexes, cword_indexes, Y, CY
    CNN = CNN.to(device)


    print("training....")


    #torch.save(ATLINEAR.state_dict(), "ATLINEAR_untrained.pickle")

    train(CNN, 30, 0.00001, lda_data, clda_data, meta_data, cmeta_data, word_indexes, cword_indexes, Y, CY)

    #torch.save(ATLINEAR.state_dict(), "ATLINEAR_trained.pickle")
    main()
