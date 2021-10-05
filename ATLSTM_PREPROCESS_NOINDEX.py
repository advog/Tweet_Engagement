
import pickle
import numpy as np

def main():

    #######################################GET_FEATURES_GET_FEATURES_GET_FEATURES##########################################
    pickle_in = open('tweet_list_full.pickle', 'rb')
    fin = pickle.load(pickle_in)
    pickle_in.close()

    tweetList = fin
    print(len(tweetList))

    FINAL = []
    text_data=[]

    for tweets in tweetList:

        tmp = tweets[3].split()

        HASHBOOL = False
        RTBOOL = False
        ATBOOL = False
        IMAGEBOOL = False
        RESBOOL = False
        hashtags = []
        ats = []
        imageURL = "null"

        ID = tweets[0]
        RTS = tweets[1]
        FAVS = tweets[2]
        TEXT = tweets[3]

        if tmp[0] == b"RT":
            RTBOOL = True

        if tmp[0].find(b'#')>-1:
            RESBOOL = True

        for word in tmp:
            if(word.find(b'#')>-1):
                HASHBOOL = True

            if word.find(b'@') == 0 and len(word) > 1:
                ATBOOL = True

        if (tmp[len(tmp)-1].find(b'https') > -1):
            IMAGEBOOL = True
            imageURL = word
            # print(imageURL)

        if (RTBOOL == False):
            FINAL.append([ID, RTS, FAVS, TEXT, HASHBOOL, RTBOOL, ATBOOL, IMAGEBOOL, RESBOOL, hashtags, ats, imageURL])
            text_data.append((tweets[3]))


    X = []
    Y = []
    DOCLIST = []

    for num in range(len(FINAL)):

        if(FINAL[num][6] == False):
            X.append(FINAL[num][4:8])
            Y.append(FINAL[num][1]+FINAL[num][2])
            DOCLIST.append(FINAL[num][3])

        print(len(X))

    #############################################################################################################################

    from collections import OrderedDict
    from string import punctuation

    #remove = punctuation
    #word = word.replace('#', '').replace('@', '').replace('/', '').replace('{', '')
    #word = word.replace('\n', '').replace('  ', ' ').replace('   ', ' ')

    ##########################################################LDA_LDA_LDA_LDA################################################################

    proc = []

    with_at_hash = []

    # remove hashtags, ats, hhtps, rts and split
    for text in DOCLIST:

        split = text.lower().split()
        ptext = []
        notptext = []
        for word in split:
            if (word.find(b'http') == -1 and word.find(
                    b'rt') == -1 and word.find(b'&') == -1):
                notptext.append(word.decode('ascii'))

            if (word.find(b'#') == -1 and word.find(b'@') == -1 and word.find(b'http') == -1 and word.find(
                    b'rt') == -1 and word.find(b'&') == -1):
                ptext.append(word.decode('ascii'))

        proc.append(' '.join(ptext))
        with_at_hash.append(' '.join(notptext))



    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.wordnet import WordNetLemmatizer
    import string
    import numpy

    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    exclude_with = set(string.punctuation.replace('#', '').replace('@', ''))

    lemma = WordNetLemmatizer()

    def clean_clean(doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    def clean_with(doc):
        punc_free = ''.join(ch for ch in doc if ch not in exclude_with)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    doc_clean = [clean_clean(doc).split() for doc in proc]
    doc_clean_with_at_hash = [clean_with(doc).split() for doc in with_at_hash]

    print(DOCLIST[5])
    print(doc_clean_with_at_hash[5])



    import gensim
    from gensim import corpora
    dictionary = corpora.Dictionary(doc_clean)

    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    Lda = gensim.models.ldamodel.LdaModel
    print("train")
    # Running and Training LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix[:3000], num_topics=20, id2word=dictionary, alpha='auto', eval_every=5, passes=50)
    lda_dat = []

    for i in range(0, len(doc_term_matrix)):
        # vectors.append(ldamodel[doc_term_matrix[i]])
        tmp = (ldamodel.get_document_topics(doc_term_matrix[i], minimum_probability=0.0))
        vector = []
        for scal in tmp:
            vector.append(scal[1])
        lda_dat.append(vector)

    print((lda_dat[1]))

    pickle_out = open("ldamodel.pickle", "wb")
    pickle.dump(ldamodel, pickle_out)
    pickle_out.close()
    pickle_out = open("doc_term_matrix.pickle", "wb")
    pickle.dump(doc_term_matrix, pickle_out)
    pickle_out.close()


########################################################## PERCENTILES #############################################################

    Y = np.array(Y)

    YP1 = np.percentile(Y,20)
    YP2 = np.percentile(Y, 40)
    YP3 = np.percentile(Y, 60)
    YP4 = np.percentile(Y, 80)


    Y = Y.tolist()

    for i in range(len(Y)):
        tmp = [0]*4
        if (Y[i] > YP1):
            tmp[0]=1
        if (Y[i] > YP2):
            tmp[1]=1
        if (Y[i] > YP3):
            tmp[2]=1
        if (Y[i] > YP4):
            tmp[3]=1
        Y[i] = tmp

#################################################word vecs#####################

  
    #print(reviews_ints)

    data_ints = []
    #for review in reviews_ints:
        #data_ints.append([vocab_to_int[word] for word in review])

    model = gensim.models.KeyedVectors.load_word2vec_format('D:\\GLOVE\\twitter_word2vec.txt')

    for l in range(len(doc_clean)):
        lego = []
        for word in doc_clean[l]:
            if word in model.wv.vocab:
                lego.append(model.wv[word])
            else:
                lego.append(model.wv["tomato"])

        data_ints.append(lego)

    print(len(data_ints))
    print(len(X))

##########################################N-Grams#########################################################

    with_array = []

    for a in doc_clean_with_at_hash:
        tboy = ''
        for b in a:
            tboy = tboy + ' ' +b
        with_array.append(tboy)

    print(len(with_array))

    from sklearn.feature_extraction.text import CountVectorizer

    vecs = CountVectorizer(ngram_range=(1, 3), min_df=10)
    trained = vecs.fit_transform(with_array).toarray().tolist()

    print(X[0])
    for i in range(len(X)):
        #print(trained[i])
        X[i] = X[i] + (trained[i])

    print(len(X[0]))

    #2166 in length


##########################################TO Tensor then Pickle###########################################


    #print(doc_clean[1])
    # print(lda_data[1])
    #print(X[1])


    import torch

    TRsplit = int(len(X)*0.75)
    CVsplit = int(len(X)*0.90)

    TY = Y[CVsplit:]
    CY = Y[TRsplit:CVsplit]
    Y = Y[:TRsplit]

    tmeta_data = X[CVsplit:]
    cmeta_data = X[TRsplit:CVsplit]
    meta_data = X[:TRsplit]

    tword_index = data_ints[CVsplit:]
    cword_index = data_ints[TRsplit:CVsplit]
    word_index = data_ints[:TRsplit]

    tlda_data = lda_dat[CVsplit:]
    clda_data = lda_dat[TRsplit:CVsplit]
    lda_data = lda_dat[:TRsplit]

    tlda_data = torch.Tensor(tlda_data)
    clda_data = torch.Tensor(clda_data)
    lda_data = torch.Tensor(lda_data)

    tword_indexes = []
    for i in range(len(tword_index)):
        tword_indexes.append(torch.Tensor(tword_index[i]))

    cword_indexes = []
    for i in range(len(cword_index)):
        cword_indexes.append(torch.Tensor(cword_index[i]))

    word_indexes = []
    for i in range(len(word_index)):
        word_indexes.append(torch.Tensor(word_index[i]))

    tmeta_data = torch.Tensor(tmeta_data)
    cmeta_data = torch.Tensor(cmeta_data)
    meta_data = torch.Tensor(meta_data)

    TY = torch.Tensor(TY)
    CY = torch.Tensor(CY)
    Y = torch.Tensor(Y)



    pickle_out = open("clda_data.pickle", "wb")
    pickle.dump(clda_data, pickle_out)
    pickle_out.close()
    pickle_out = open("lda_data.pickle", "wb")
    pickle.dump(lda_data, pickle_out)
    pickle_out.close()
    pickle_out = open("cmeta_data.pickle", "wb")
    pickle.dump(cmeta_data, pickle_out)
    pickle_out.close()
    pickle_out = open("meta_data.pickle", "wb")
    pickle.dump(meta_data, pickle_out)
    pickle_out.close()
    pickle_out = open("word_indexes.pickle", "wb")
    pickle.dump(word_indexes, pickle_out)
    pickle_out.close()
    pickle_out = open("cword_indexes.pickle", "wb")
    pickle.dump(cword_indexes, pickle_out)
    pickle_out.close()
    pickle_out = open("Y.pickle", "wb")
    pickle.dump(Y, pickle_out)
    pickle_out.close()
    pickle_out = open("CY.pickle", "wb")
    pickle.dump(CY, pickle_out)
    pickle_out.close()
    #pickle_out = open("vocab_size.pickle", "wb")
    #pickle.dump(vocab_size, pickle_out)
    #pickle_out.close()

    pickle_out = open("tlda_data.pickle", "wb")
    pickle.dump(tlda_data, pickle_out)
    pickle_out.close()
    pickle_out = open("tmeta_data.pickle", "wb")
    pickle.dump(tmeta_data, pickle_out)
    pickle_out.close()
    pickle_out = open("tword_indexes.pickle", "wb")
    pickle.dump(tword_indexes, pickle_out)
    pickle_out.close()
    pickle_out = open("TY.pickle", "wb")
    pickle.dump(TY, pickle_out)
    pickle_out.close()

    #pickle_out = open("vocab_to_int.pickle", "wb")
    #pickle.dump(vocab_to_int, pickle_out)
    #pickle_out.close()
main()
