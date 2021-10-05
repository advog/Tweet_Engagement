from string import punctuation
import pickle
import torch
from ATLINEAR import ATLINEAR
from collections import OrderedDict

if __name__ == "__main__":
    pickle_in = open('vocab_to_int.pickle', 'rb')
    vocab_to_int = pickle.load(pickle_in)
    pickle_in.close()
    #pickle_in = open('ldamodel', 'rb')
    #ldamodel = pickle.load(pickle_in)
    #pickle_in.close()
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
    pickle_in = open('vocab_size.pickle', 'rb')
    vocab_size = pickle.load(pickle_in)
    pickle_in.close()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tlda_data = tlda_data.to(device)
    tmeta_data = tmeta_data.to(device)
    TY = TY.to(device)

    encoder_dim = 128
    ordinal_classes = TY.shape[1]
    full_feature_size = (tlda_data.shape[1] + tmeta_data.shape[1] + 2 * encoder_dim)
    attentioner_dim = 64
    embedding_dim = 256


    tweet = ".@FDATobacco is at #APHA2019 located at booth #1327. Come talk with us about the work you are doing to combat tobacco use in your community and find out what FDA is doing to help your efforts."

    print(punctuation)
    remove = punctuation
    remove = remove.replace('#','').replace('@','').replace('/','').replace('{','')

    tweet = tweet.lower()  # lowercase, standardize
    clean = ''.join([c for c in tweet if c not in remove])
    # split by new lines and spaces
    clean = clean.replace('\n','').replace('  ',' ').replace('   ',' ')

    # create a list of words
    words = tweet.split()

    tmp = []
    for word in tweet.split(' '):
        word = word.replace(' ','')
        if(word == ''):
            1+1
        elif ('/' in word):
            tmp.append('link_to_external_source')
        else:
            tmp.append(word)
        #print(tmp)

    index = []
    for word in tmp:
        if word in vocab_to_int:
            index.append(vocab_to_int[word])
        else:
            index.append(vocab_to_int['unkkown_word'])

    #lda_vector = (ldamodel.get_document_topics(doc_term_matrix[i], minimum_probability=0.0))

    model = ATLINEAR(vocab_size + 1, embedding_dim, encoder_dim, attentioner_dim, full_feature_size,ordinal_classes).to(device)
    model.load_state_dict(torch.load("ATLSTM_trained.pickle"))
    model = model.to(device)
    model.eval()

    #HASHBOOL, RTBOOL, ATBOOL, IMAGEBOOL,
    nmeta_data = torch.tensor([0,0,1,1]+[0]*530, dtype= torch.float).to(device)
    nlda_data = torch.tensor([0]*20, dtype= torch.float).to(device)
    word_index = torch.LongTensor(index)

    ch_encoder = model.Encoder.init_hidden()
    ch_attentioner = model.Attentioner.init_hidden()

    output = model(word_index, nmeta_data, nlda_data, 0, ch_encoder, ch_attentioner)
    attention = model.att(word_index, nmeta_data, nlda_data, 0, ch_encoder, ch_attentioner)

    print(output)
    print()

    attention_scalars = []
    for i in range(len(attention)):
        attention_scalars.append(attention[i].item())

    for i in range(len(attention_scalars)):
        print(tmp[i], attention_scalars[i])
