import pickle
import torch
from LSTMMACPOOL import LSTMMACPOOL
from ATLINEAR import ATLINEAR
from ATLSTM import ATLSTM
from CNN import CNN
from LOGISTIC import LOGISTIC
import numpy
from ORDINAL import ANN


pickle_in = open('tlda_data.pickle', 'rb')
tlda_data = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open('tmeta_data.pickle', 'rb')
tmeta_data = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open('tword_indexes.pickle', 'rb')
tword_indexes = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open('TY.pickle', 'rb')
TY = pickle.load(pickle_in)
pickle_in.close()



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tlda_data = tlda_data.to(device)
tmeta_data = tmeta_data.to(device)
TY = TY.to(device)

encoder_dim = 128
ordinal_classes = TY.shape[1]
full_feature_size = (tlda_data.shape[1] + tmeta_data.shape[1] + 2*encoder_dim)
#full_feature_size = (tlda_data.shape[1] + tmeta_data.shape[1])
attentioner_dim = 64
embedding_dim = 200
cnn_channels = 32
ngram_s = 5

#address = "CNN_Embed:" + str(embedding_dim) + "_" + str(cnn_channels) + "_" + str(ngram_s) + ".pickle"
address = "B_ATLINEAR_Embed:"+ str(embedding_dim) +"_"+ str(encoder_dim) +".pickle"
#address = "LOGISTIC.pickle"
#address = "ANN.pickle"
print(address)

model = ATLINEAR(1,embedding_dim,encoder_dim,attentioner_dim,full_feature_size,ordinal_classes).to(device)
#model = CNN(vocab_size + 1, embedding_dim, full_feature_size, ordinal_classes, cnn_channels, ngram_s).to(device)
#model = LOGISTIC(full_feature_size, ordinal_classes)
#model = ANN(full_feature_size, ordinal_classes)

model.load_state_dict(torch.load(address))
model = model.to(device)
model.eval()


def evaluate(threshold):
    total = 0

    ch_encoder = model.Encoder.init_hidden()
    ch_attentioner = model.Attentioner.init_hidden()

    numclasses = 5

    classdiffs = [0] * numclasses
    classcount = [0] * numclasses
    classmaxwrong = [4, 3, 2, 3, 4]
    N = len(TY)

    for i in range(len(TY)):
        if(len(tword_indexes[i]) >= ngram_s):
            ch_encoder = tuple([each.data for each in ch_encoder])
            ch_attentioner = tuple([each.data for each in ch_attentioner])

            # print(cword_indexes[i])

            #output = model(tword_indexes[i], tmeta_data[i], tlda_data[i], TY[i])
            output = model(tword_indexes[i], tmeta_data[i], tlda_data[i], TY[i], ch_encoder, ch_attentioner)
            #output = model(tmeta_data[i], tlda_data[i])

            output = output.cpu().detach().numpy()

            output_ordinal = 0
            for bo in range(len(output)):
                if (output[bo] > threshold):
                    output_ordinal += 1


            expected = TY[i].cpu().detach().numpy()
            expected_ordinal = int(numpy.sum(expected))

            classcount[expected_ordinal] += 1
            classdiffs[expected_ordinal] += abs(expected_ordinal - output_ordinal)


    total = 0
    # print(classcount)
    # print(classdiffs)
    classadjusteddiffs = [0] * numclasses
    for cl in range(len(classcount)):
        classadjusteddiffs[cl] = classdiffs[cl] / (classcount[cl] * classmaxwrong[cl])

    # print(classadjusteddiffs)
    total = sum(classadjusteddiffs)
    AMMAE = (1 - total / len(classadjusteddiffs)) * 100

    print(AMMAE)

evaluate(0.4)
evaluate(0.45)
evaluate(0.5)
evaluate(0.57)
evaluate(0.6)
evaluate(0.65)

