import pickle
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class Embeder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Embeder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)

    def forward(self, inputs):
        #inputs tensor of shape (sequence_length, 1)
        inputs = inputs.to(device)
        output = self.embedding_layer(inputs)
        return output
        # outputs tensor of shape (1,sequence_length, embedding_dim)

class Encoder(nn.Module):
    #feature size should be the embedding_dim of ^
    #2*hidden_size is the output size of one LSTMcell
    def __init__(self, feature_size, hidden_size, bidirectional=True, batch_first = True):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = feature_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, bidirectional=bidirectional, batch_first=batch_first)

    def forward(self, inputs, hidden):
        #inputs tensor of shape (batch_size = 1, seq_length, embedding_dim)
        output, hidden = self.lstm(inputs.view(1, -1, self.input_size), hidden)
        # outputs tensor of shape (1,sequence_length, 2*hidden_dim)
        return output

    def init_hidden(self):
        return (torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size).to(device),
                torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size).to(device))


class Combiner(nn.Module):
    #combines attention with hidden values of encoder
    def __init__(self):
        super(Combiner, self).__init__()


    def forward(self, encoded_sequence):
        #print(encoded_sequence.shape)
        output = torch.max(encoded_sequence, dim = 1)[0].view(-1)
        return output


class ANNer(nn.Module):
    def __init__(self, feature_size, ordinal_classes):
        super().__init__()
        self.feature_size = feature_size
        self.ordinal_classes = ordinal_classes

        self.linear = nn.Linear(self.feature_size, self.ordinal_classes)
        self.sigy = nn.Sigmoid()

    def forward(self, inputs):
        #inputs tensor of shape (metadata + LDA + specific_words + encoding_dim, 1)
        output = self.linear(inputs)
        output = self.sigy(output)
        #outpurs tensor of shape (ordinal_classes, 1)
        return output

class Ordinal_loss(torch.nn.Module):
    #moddified binary cross entropy
    def __init__(self):
        super(Ordinal_loss, self).__init__()

    def forward(self, predicted, actual):
        BCE = torch.add(    torch.mul(actual,torch.log(predicted)) ,   torch.mul(  torch.sub(1,actual)   , torch.log(torch.sub(1,predicted))  ))
        return torch.neg(torch.sum(BCE))


class LSTMMACPOOL(torch.nn.Module):
    #moddified binary cross entropy
    def __init__(self, vocab_size, embedding_dim, encoder_dim, attentioner_dim, full_feature_size, ordinal_classes):
        super(LSTMMACPOOL, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        self.attentioner_dim = attentioner_dim
        self.full_feature_size = full_feature_size
        self.ordinal_classes = ordinal_classes


        self.Embeder = Embeder(self.vocab_size, self.embedding_dim)
        self.Encoder = Encoder(self.embedding_dim, self.encoder_dim)
        self.Combiner = Combiner()
        self.ANNer = ANNer(self.full_feature_size, self.ordinal_classes)

    def forward(self, word_indexes, meta_data, lda_data, actual, h_encoder):
        embedded_sequence = self.Embeder(word_indexes)
        encoded_sequence = self.Encoder(embedded_sequence, h_encoder)
        #print(encoded_sequence.shape)
        #print(attention_values)
        attentioned_encoded_rep = self.Combiner(encoded_sequence)
        #print(attentioned_encoded_rep)
        #print(lda_data)
        output = self.ANNer(torch.cat((attentioned_encoded_rep,meta_data,lda_data)))
        return output

def train(net, epochs, lr, lda_data, clda_data, meta_data, cmeta_data, word_indexes, cword_indexes, Y, CY):
    #inputs ATLSTM network, epochs, learning_rate, and the data split and in tensor form (.cuda)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    criterion = Ordinal_loss()

    mean_losses = []

    epoch_loss = 99
    last_epoch_loss = 100

    for e in range(epochs):

        val_losses = []

        ch_encoder = net.Encoder.init_hidden()

        net.eval()

        for i in range(len(CY)):

            ch_encoder = tuple([each.data for each in ch_encoder])

            #print(cword_indexes[i])

            output = net(cword_indexes[i], cmeta_data[i], clda_data[i], CY[i], ch_encoder)

            loss = criterion(output, CY[i])
            val_losses.append(loss.item())

        last_epoch_loss = epoch_loss
        epoch_loss = float(np.mean(val_losses))
        print(epoch_loss)

        if last_epoch_loss < epoch_loss:
            return

        h_encoder = net.Encoder.init_hidden()

        net.train()

        for i in range(len(Y)):

            h_encoder = tuple([each.data for each in h_encoder])

            net.zero_grad()
            output = net(word_indexes[i], meta_data[i], lda_data[i], Y[i], h_encoder)

            #print(output)

            loss = criterion(output, Y[i])
            loss.backward()
            optimizer.step()

    return mean_losses

# stuff to run always here such as class/def
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

    Y = Y.to(device)
    CY = CY.to(device)

    print(Y.shape)

    encoder_dim = 128
    ordinal_classes = Y.shape[1]
    full_feature_size = (lda_data.shape[1] + meta_data.shape[1] + 2 * encoder_dim)
    attentioner_dim = 64
    embedding_dim = 256

    # vocab_size, embedding_dim, encoder_dim, attentioner_dim, full_feature_size, ordinal_classes
    ATLSTM = LSTMMACPOOL(vocab_size + 1, embedding_dim, encoder_dim, attentioner_dim, full_feature_size, ordinal_classes)

    # net, epochs, lr, lda_data, clda_data, meta_data, cmeta_data, word_indexes, cword_indexes, Y, CY
    ATLSTM = ATLSTM.to(device)

    """
    lda_data = torch.Tensor([[0.01, 0.04, 0.04, 0.05]]).cuda()
    clda_data = torch.Tensor([[0.01, 0.04, 0.04, 0.05]]).cuda()
    meta_data = torch.Tensor([[1, 1, 0, 1]]).cuda()
    cmeta_data = torch.Tensor([[1, 1, 0, 1]]).cuda()
    word_indexes = torch.LongTensor([[3,2,3,4]]).cuda()
    cword_indexes = torch.LongTensor([[4,2,3,4]]).cuda()
    Y = torch.Tensor([[0, 1]]).cuda()
    CY = torch.Tensor([[0, 1]]).cuda()
    """

    print("training....")

    train(ATLSTM, 1, 0, lda_data, clda_data, meta_data, cmeta_data, word_indexes, cword_indexes, Y, CY)

    torch.save(ATLSTM.state_dict(), "ATPOOL_untrained.pickle")


    train(ATLSTM, 30, 0.00001, lda_data, clda_data, meta_data, cmeta_data, word_indexes, cword_indexes, Y, CY)

    torch.save(ATLSTM.state_dict(), "ATPOOL_trained.pickle")
    main()
