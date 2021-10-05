import pickle
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib

import gensim



class Encoder(nn.Module):
    #feature size should be the embedding_dim of ^
    #2*hidden_size is the output size of one LSTMcell
    def __init__(self, feature_size, hidden_size, bidirectional=True, batch_first = True):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = feature_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, bidirectional=bidirectional, batch_first=batch_first, dropout=0.4)

    def forward(self, inputs, hidden):
        #inputs tensor of shape (batch_size = 1, seq_length, embedding_dim)
        inputs = inputs.to(device)
        output, hidden = self.lstm(inputs.view(1, -1, self.input_size), hidden)
        # outputs tensor of shape (1,sequence_length, 2*hidden_dim)
        return output

    def init_hidden(self):
        return (torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size).to(device),
                torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size).to(device))


class Attentioner(nn.Module):
    #feature_size should equal 2*hidden_size of ^
    def __init__(self, feature_size, hidden_size, bidirectional = True, batch_first = True, bias = False):
        super(Attentioner, self).__init__()
        self.input_size = feature_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        #self.lstm = nn.LSTM(self.input_size, self.hidden_size, bidirectional=bidirectional, batch_first=batch_first)
        self.matrix = nn.Linear(self.input_size,self.hidden_size)
        print(self.input_size)
        self.tany = nn.Tanh()
        self.vector = nn.Linear(self.hidden_size,1,bias=False)

    def forward(self, inputs, hidden):

        #inputs tensor of shape (1,sequence_length, 2*hidden_dim_of_encoder)
        #output, hidden = self.lstm(inputs.view(1, -1, self.input_size), hidden)
        #outputs tensor of shape (sequence_length, 2*hiddent_dim)
        output = self.matrix(inputs)
        output = self.tany(output)
        output = self.vector(output)
        #outputs tensor of shape (sequence_length, 1)
        output = F.softmax(output,dim =1).view(-1,1)
        #print(output.shape)
        # outputs tensor of shape (1,sequence_length, 1) but softmaxed

        return output

    def init_hidden(self):
        return (torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size).to(device),
                torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size).to(device))



class Combiner(nn.Module):
    #combines attention with hidden values of encoder
    def __init__(self):
        super(Combiner, self).__init__()

        self.tany = nn.Tanh()

    def forward(self, encoded_sequence, attention_values):
        #print("enc_seq: ",encoded_sequence.shape)
        #print("att_val: ",attention_values.shape)
        #inputs encoded_sequence tensor of shape (sequence_length, encoding_dim)
        #inputs attention_values tensor of shape (sequence_length, 1)
        output = torch.mul(attention_values, encoded_sequence)
        #print("out befor: ", output.shape)
        #outputs tensor of shape (sequence_length, encoding dim)
        output = torch.sum(output, dim =1).view(-1)
        output = self.tany(output)
        #print("out: ", output.shape)
        #outputs tensor of shape (encoding_dim)
        return output


class ANNer(nn.Module):
    def __init__(self, feature_size, ordinal_classes):
        super().__init__()
        self.feature_size = feature_size
        print(feature_size)
        #self.ordinal_classes = ordinal_classes
        self.ordinal_classes = 4

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
        #print(predicted)
        #print(BCE)
        #print(torch.neg(torch.sum(BCE)))
        return torch.neg(torch.sum(BCE))


class ATLINEAR(torch.nn.Module):
    #moddified binary cross entropy
    def __init__(self, vocab_size, embedding_dim, encoder_dim, attentioner_dim, full_feature_size, ordinal_classes):
        super(ATLINEAR, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        self.attentioner_dim = attentioner_dim
        self.full_feature_size = full_feature_size
        self.ordinal_classes = ordinal_classes


        #self.Embeder = Embeder(self.vocab_size, self.embedding_dim)
        self.Encoder = Encoder(self.embedding_dim, self.encoder_dim)
        self.Attentioner = Attentioner(2*self.encoder_dim, self.attentioner_dim)
        self.Combiner = Combiner()
        self.ANNer = ANNer(self.full_feature_size, self.ordinal_classes)

    def forward(self, word_indexes, meta_data, lda_data, actual, h_encoder, h_attentioner):
        #embedded_sequence = self.Embeder(word_indexes)
        encoded_sequence = self.Encoder(word_indexes, h_encoder)
        #print(encoded_sequence.shape)
        attention_values = self.Attentioner(encoded_sequence, h_attentioner)
        #print(attention_values)
        attentioned_encoded_rep = self.Combiner(encoded_sequence, attention_values)
        #output = attention_encoded_rep
        #print(attentioned_encoded_rep)
        #print(lda_data)
        output = self.ANNer(torch.cat((attentioned_encoded_rep,torch.mul(meta_data,1),torch.mul(lda_data,1))))


        return output

    def att(self, word_indexes, meta_data, lda_data, actual, h_encoder, h_attentioner):
        embedded_sequence = self.Embeder(word_indexes)
        encoded_sequence = self.Encoder(embedded_sequence, h_encoder)
        # print(encoded_sequence.shape)
        attention_values = self.Attentioner(encoded_sequence, h_attentioner)
        return attention_values


def train(save_address,net, epochs, lr, lda_data, clda_data, meta_data, cmeta_data, word_indexes, cword_indexes, Y, CY):
    #inputs ATLINEAR network, epochs, learning_rate, and the data split and in tensor form (.cuda)
    net.train()
    #optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer = torch.optim.Adagrad(net.parameters(), lr=lr)
    criterion = Ordinal_loss()

    epoch_loss = 99
    lowest_epoch_loss = 100

    for e in range(epochs):

        val_losses = []

        ch_encoder = net.Encoder.init_hidden()
        ch_attentioner = net.Attentioner.init_hidden()

        net.eval()

        for i in range(len(CY)):

            ch_encoder = tuple([each.data for each in ch_encoder])
            ch_attentioner = tuple([each.data for each in ch_attentioner])

            #print(cword_indexes[i])

            output = net(cword_indexes[i], cmeta_data[i], clda_data[i], CY[i], ch_encoder, ch_attentioner)

            loss = criterion(output, CY[i])
            val_losses.append(loss.item())

        epoch_loss = float(np.mean(val_losses))
        print(epoch_loss)

        if epoch_loss < lowest_epoch_loss:
            lowest_epoch_loss = epoch_loss
            print(save_address)
            torch.save(net.state_dict(), save_address)


        h_encoder = net.Encoder.init_hidden()
        h_attentioner = net.Attentioner.init_hidden()

        net.train()

        for i in range(len(Y)):

            h_encoder = tuple([each.data for each in h_encoder])
            h_attentioner = tuple([each.data for each in h_attentioner])

            net.zero_grad()
            output = net(word_indexes[i], meta_data[i], lda_data[i], Y[i], h_encoder, h_attentioner)

            #print(output)

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
    #pickle_in = open('vocab_size.pickle', 'rb')
    #vocab_size = pickle.load(pickle_in)
    #pickle_in.close()

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

    attentioner_dim = 64


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

    # train(ATLINEAR, 1, 0, lda_data, clda_data, meta_data, cmeta_data, word_indexes, cword_indexes, Y, CY)
    # torch.save(ATLINEAR.state_dict(), "ATLINEAR_untrained.pickle")

    print("training....")

    encoder_dim = 128
    embedding_dim = 200
    address = "B_ATLINEAR_Embed:" + str(embedding_dim) + "_" + str(encoder_dim) + ".pickle"
    print(address)

    print(meta_data.shape)
    full_feature_size = (lda_data.shape[1] + meta_data.shape[1] + 2 * encoder_dim)
    # vocab_size, embedding_dim, encoder_dim, attentioner_dim, full_feature_size, ordinal_classes
    ATLINEAR = ATLINEAR(1, embedding_dim, encoder_dim, attentioner_dim, full_feature_size, ordinal_classes)
    # net, epochs, lr, lda_data, clda_data, meta_data, cmeta_data, word_indexes, cword_indexes, Y, CY
    ATLINEAR = ATLINEAR.to(device)
    train(address,ATLINEAR, 100, 0.0005, lda_data, clda_data, meta_data, cmeta_data, word_indexes, cword_indexes, Y, CY)


    import winsound

    duration = 3000  # milliseconds
    freq = 880  # Hz
    winsound.Beep(freq, duration)

    main()


#[1,1,1,1]
#[0,0,0,0]
#[1,1,0,0]