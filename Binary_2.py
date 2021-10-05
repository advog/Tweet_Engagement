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
#full_feature_size = (tlda_data.shape[1] + tmeta_data.shape[1] + 2*encoder_dim)
#full_feature_size = (tlda_data.shape[1] + tmeta_data.shape[1])
full_feature_size = (tlda_data.shape[1] + tmeta_data.shape[1] +200)
attentioner_dim = 64
embedding_dim = 200
cnn_channels = 100
ngram_s = 5

address = "CNN_Embed:" + str(embedding_dim) + "_" + str(cnn_channels) + "_" + str(ngram_s) + ".pickle"
#address = "LOGISTIC.pickle"
#address = "ANN.pickle"
print(address)

#model = ATLINEAR(1,embedding_dim,encoder_dim,attentioner_dim,full_feature_size,ordinal_classes).to(device)
model = CNN(1, embedding_dim, full_feature_size, ordinal_classes, cnn_channels, ngram_s).to(device)
#model = LOGISTIC(full_feature_size, ordinal_classes)
#model = ANN(full_feature_size, ordinal_classes)

model.load_state_dict(torch.load(address))
model = model.to(device)
model.eval()


def evaluate(threshold,bar):

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(TY)):
        if(len(tword_indexes[i]) >= ngram_s):


            ebin = False
            obin = False

            output = model(tword_indexes[i], tmeta_data[i], tlda_data[i], TY[i])
            #output = model(tmeta_data[i], tlda_data[i])

            output = output.cpu().detach().numpy()

            output_ordinal = 0
            for bo in range(len(output)):
                if (output[bo] > threshold):
                    output_ordinal += 1

            expected = TY[i].cpu().detach().numpy()
            expected_ordinal = int(numpy.sum(expected))
            #print(output_ordinal)
            if(expected_ordinal > bar):
                ebin = True
            if(output_ordinal > bar):
                obin = True

            #print(str(obin) + "  " + str(ebin))

            if(ebin and obin):
                tp+=1
            if(ebin and (not obin)):
                fn+=1
            if((not ebin) and obin):
                fp+=1
            if((not ebin) and (not obin)):
                tn+=1

    print(tp)

    perc = tp / (tp + fp)
    rec = tp / (tp + fn)

    f1 = (2 * tp) / (2 * tp + fp + fn)

    print("    " + str(tp) + " " + str(fp) + " " + str(fn) + " " + str(tn) + "    " + str(f1) + "    " + str(
        rec) + "    " + str(perc))

evaluate(0.1,3)
evaluate(0.35,3)
evaluate(0.40,3)
evaluate(0.45,3)
evaluate(0.5,3)
evaluate(0.55,3)
evaluate(0.6,3)
evaluate(0.65,3)

#for param in model.parameters(LOGISTIC):
  #print(param.data)