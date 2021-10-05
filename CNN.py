import pickle
import torch
import numpy as np
from torch import nn

class Convolver(nn.Module):
    def __init__(self, channels_in, channels_out, conv_size, embedding_dim):
        super(Convolver,self).__init__()
        self.embedding_dim = embedding_dim
        self.channelse_in = channels_in
        self.channels_out = channels_out
        self.conv_size = conv_size

        self.convoluting_layer3 = nn.Conv2d(self.channelse_in, self.channels_out, (3, self.embedding_dim), stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.convoluting_layer4 = nn.Conv2d(self.channelse_in, self.channels_out, (4, self.embedding_dim), stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.convoluting_layer5 = nn.Conv2d(self.channelse_in, self.channels_out, (5, self.embedding_dim), stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

        self.rely = nn.ReLU()

    def forward(self,input):
        # inputs tensor of shape (sequence_length, embedding_dim)
        inputs = input.to(device)
        input = input.view(1,1,input.shape[0],input.shape[1])
        output1 = self.convoluting_layer3(input)
        output2 = self.convoluting_layer4(input)
        output3 = self.convoluting_layer5(input)
        output1 = self.rely(output1.view(100, -1))
        output2 = self.rely(output2.view(100, -1))
        output3 = self.rely(output3.view(100, -1))


        output = (output1, output2, output3)
        return output
        # outputs tensor of shape (channel_num, sequence length - (ngram size-1))

class Combiner(nn.Module):
    #combines attention with hidden values of encoder
    def __init__(self):
        super(Combiner, self).__init__()

    def forward(self, encoded_sequence):
        # inputs tensor of shape (sequence length - (ngram size-1), 1, channel num)
        output1 = torch.max(encoded_sequence[0], dim = 1)[0].view(-1)
        output2 = torch.max(encoded_sequence[1], dim=1)[0].view(-1)
        output3 = torch.max(encoded_sequence[2], dim=1)[0].view(-1)
        output = torch.cat((output1, output2, output3), 0)
        return output
    #outputs tensor of shape (chanel num)

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


class CNN(torch.nn.Module):
    #moddified binary cross entropy
    def __init__(self, vocab_size, embedding_dim, full_feature_size, ordinal_classes, cnn_channels, ngram_length):
        super(CNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.full_feature_size = full_feature_size
        self.ordinal_classes = ordinal_classes
        self.cnn_channels = cnn_channels
        self.ngram_length = ngram_length

        self.Convoluter = Convolver(1,cnn_channels, ngram_length,embedding_dim)
        self.Combiner = Combiner()
        #print(self.full_feature_size+self.ngram_length)
        self.ANNer = ANNer(self.full_feature_size + self.cnn_channels, self.ordinal_classes)

    def forward(self, word_indexes, meta_data, lda_data, actual):
        word_indexes = word_indexes.to(device)
        convolutions = self.Convoluter(word_indexes)
        conv_maxes = self.Combiner(convolutions)
        #print(conv_maxes)
        output = self.ANNer(torch.cat((conv_maxes,meta_data,lda_data)))
        return output


def train(save_address, net, epochs, lr, lda_data, clda_data, meta_data, cmeta_data, word_indexes, cword_indexes, Y, CY):
    # inputs ATLINEAR network, epochs, learning_rate, and the data split and in tensor form (.cuda)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    criterion = Ordinal_loss()

    epoch_loss = 100
    lowest_epoch_loss = 99

    for e in range(epochs):

        val_losses = []

        net.eval()

        for i in range(len(CY)):

            # print(cword_indexes[i])
            if(len(cword_indexes[i]) > 7):
                output = net(cword_indexes[i], cmeta_data[i], clda_data[i], CY[i])

                loss = criterion(output, CY[i])
                val_losses.append(loss.item())

        epoch_loss = float(np.mean(val_losses))
        print(epoch_loss)

        if epoch_loss < lowest_epoch_loss:
            lowest_epoch_loss = epoch_loss
            print(save_address)
            torch.save(net.state_dict(), save_address)

        net.train()

        for i in range(len(Y)):

            if (len(word_indexes[i]) > 7):
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

    ngram_s = 5
    ordinal_classes = Y.shape[1]
    print(lda_data.shape[1] + meta_data.shape[1])
    full_feature_size = (lda_data.shape[1] + meta_data.shape[1] + 100*2)
    embedding_dim = 200
    cnn_channels = 100

    # vocab_size, embedding_dim, encoder_dim, attentioner_dim, full_feature_size, ordinal_classes
    CNN = CNN(1, embedding_dim, full_feature_size, ordinal_classes, cnn_channels, ngram_s)

    # net, epochs, lr, lda_data, clda_data, meta_data, cmeta_data, word_indexes, cword_indexes, Y, CY
    CNN = CNN.to(device)


    print("training....")


    #torch.save(ATLINEAR.state_dict(), "ATLINEAR_untrained.pickle")

    address = "CNN_Embed:" + str(embedding_dim) + "_" + str(cnn_channels) + "_" + str(ngram_s) + ".pickle"
    print(address)

    train(address, CNN, 150, 0.000005, lda_data, clda_data, meta_data, cmeta_data, word_indexes, cword_indexes, Y, CY)


    torch.save(CNN.state_dict(), address)

    import winsound

    duration = 500  # milliseconds
    winsound.Beep(370 * 2, duration)
    winsound.Beep(247 * 2, duration)
    winsound.Beep(196 * 2, duration)
    winsound.Beep(156 * 2, duration)
    winsound.Beep(123 * 2, duration)
    winsound.Beep(82 * 2, duration)

    #torch.save(ATLINEAR.state_dict(), "ATLINEAR_trained.pickle")
    main()


#[1,1,1,1]
#[0,0,0,0]
#[1,1,0,0]