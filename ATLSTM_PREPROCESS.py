import json
import pickle
import numpy as np

# read json data and return as a list of list


def main():

    #######################################GET_FEATURES_GET_FEATURES_GET_FEATURES##########################################
    pickle_in = open('tweet_list_full.pickle', 'rb')
    fin = pickle.load(pickle_in)
    pickle_in.close()


    tweetList = fin

    print(len(tweetList))


    allWords = []
    allHashtags = []
    allAted = []
    FINAL = []

    text_data=[]

    for tweets in tweetList:
        text_data.append((tweets[3]))
        tmp = tweets[3].split()

        ID = 0
        RTS = 0
        FAVS = 0
        TEXT = ""
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
            allWords.append(word.lower())

            if(word.find(b'#')>-1):
                hashtag = word.lower()
                HASHBOOL = True
                allHashtags.append(hashtag)
                hashtags.append(hashtag)


            if word.find(b'@') == 0 and len(word) > 1:
                ated = word.lower()
                ATBOOL = True
                allAted.append(ated)
                ats.append(word)

        if (tmp[len(tmp)-1].find(b'https') > -1):
            IMAGEBOOL = True
            imageURL = word
            # print(imageURL)

        FINAL.append([ID, RTS, FAVS, TEXT, HASHBOOL, RTBOOL, ATBOOL, IMAGEBOOL, RESBOOL, hashtags, ats, imageURL])


    from collections import Counter
    HashCount = Counter(allHashtags).most_common()[:20]
    AtCount = Counter(allAted).most_common()[:10]
    WordCount = Counter(allWords).most_common()[:500]

    ##X[] = HASHBOOL, RTBOOL, ATBOOL, IMAGEBOOL, HASHVEC, ATVEC, WORDVEC
    ##Y1[] = RTVEC
    ##Y2[] = FAVVEC

    X = []
    Y1 = []
    Y2 = []
    DOCLIST = []

    for num in range(len(FINAL)):
        HASHVEC = [False] * 20
        ATVEC = [False] * 10
        WORDVEC = [False] * 500

        split = FINAL[num][3].split()

        for word in split:
            for i in range(20):
                if word == HashCount[i][0]:
                    HASHVEC[i] = True
            for i in range(10):
                if word == AtCount[i][0]:
                    ATVEC[i] = True
            for i in range(500):
                if word == WordCount[i][0]:
                    WORDVEC[i] = True



        if(FINAL[num][6] == False):
            #X.append(FINAL[num][4:8])
            X.append(FINAL[num][4:8]+HASHVEC+ATVEC+WORDVEC)
            Y1.append(FINAL[num][1])
            Y2.append(FINAL[num][2])

        DOCLIST.append(FINAL[num][3])

        print(len(X))

    ##########################################################LDA_LDA_LDA_LDA################################################################


    proc = []

    # remove hashtags, ats, hhtps, rts an split
    for text in DOCLIST:

        split = text.lower().split()
        ptext = []
        for word in split:
            if (word.find(b'#') == -1 and word.find(b'@') == -1 and word.find(b'http') == -1 and word.find(
                    b'rt') == -1 and word.find(b'&') == -1):
                ptext.append(word.decode('ascii'))
        proc.append(' '.join(ptext))

    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.wordnet import WordNetLemmatizer
    import string
    import numpy

    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    def clean(doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    doc_clean = [clean(doc).split() for doc in proc]


    import gensim
    from gensim import corpora

    dictionary = corpora.Dictionary(doc_clean)

    print()

    # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    #"""
    # Creating the object for LDA model using gensim library
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
    #"""

    pickle_out = open("ldamodel.pickle", "wb")
    pickle.dump(ldamodel, pickle_out)
    pickle_out.close()
    pickle_out = open("doc_term_matrix.pickle", "wb")
    pickle.dump(doc_term_matrix, pickle_out)
    pickle_out.close()

    ##########################################################AVG WORD EMBEDDINGS###################################################################

    # set the correct path to the file on your machine
    #model = gensim.models.KeyedVectors.load_word2vec_format('D:/EmbedStuff/glove.42B.300d.txt', binary=False)
    """"
    def get_mean_vector(word2vec_model, words):
        words = [word for word in words if word in word2vec_model.vocab]
        if len(words) >= 1:
            return np.mean(word2vec_model[words], axis=0).tolist()
        else:
            return np.zeros(300).tolist()

    print(doc_clean[i])

    from gensim.test.utils import datapath, get_tmpfile
    from gensim.models import KeyedVectors
    from gensim.scripts.glove2word2vec import glove2word2vec

    glove_file = datapath('D:/EmbedStuff/glove.42B.300d.txt')
    tmp_file = get_tmpfile("test_word2vec.txt")
    ayy = glove2word2vec(glove_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)

    for i in range(len(X)):
        print(doc_clean[i])
        X[i] = X[i] + (get_mean_vector(model, doc_clean[i]))

    word_vectors = []
    for i in range(len(doc_clean)):
        words = [word for word in doc_clean[i] if word in model.vocab]
        print(words)
        if(len(words) == 0):
            vecs = np.zeros([30,300])
        else:
            vecs = model[words]
            zer = np.zeros([30-vecs.shape[0],300])
            vecs = np.concatenate((zer,vecs),axis = 0)
        print(vecs)
        word_vectors.append(vecs.tolist())

    """

    ##########################################################CV_SPLIT_CV_SPLIT_CV_SPLIT + PERCENTILES PERCENTILES #############################################################

    Y2 = np.array(Y2)
    Y1 = np.array(Y1)

    Y1 = numpy.add(Y1, Y2)

    Y1P1 = np.percentile(Y1,50)
    #Y1P2 = np.percentile(Y1, 40)
    #Y1P3 = np.percentile(Y1, 60)
    #Y1P4 = np.percentile(Y1, 80)
    #Y1P5 = np.percentile(Y1, 90)


    Y1 = Y1.tolist()

    for i in range(len(Y1)):
        tmp = [0]*1
        if (Y1[i] > Y1P1):
            tmp[0]=1
        #if (Y1[i] > Y1P2):
         #   tmp[1]=1
        #if (Y1[i] > Y1P3):
         #   tmp[2]=1
        #if (Y1[i] > Y1P4):
         #   tmp[3]=1
        #if (Y1[i] > Y1P5):
            #tmp[4]=1
        Y1[i] = tmp



#################################################word INDEXES#####################

    from collections import OrderedDict
    from string import punctuation
    remove = punctuation
    remove = remove.replace('#','').replace('@','').replace('/','').replace('{','')

    # get rid of punctuation
    for t in range(len(text_data)):
        text_data[t] = text_data[t].decode('utf-8')

    reviews = ' { '.join(text_data)
    reviews = reviews.lower()  # lowercase, standardize
    clean = ''.join([c for c in reviews if c not in remove])
    # split by new lines and spaces
    clean = clean.replace('\n','').replace('  ',' ').replace('   ',' ')
    reviews_split = clean.split(' { ')

    all_text = ' '.join(reviews_split)
    # create a list of words
    words = all_text.split()


    reviews_ints = []
    for i in range(len(reviews_split)):
        tmp = []
        for word in reviews_split[i].split(' '):
            word = word.replace(' ','')
            if(word == ''):
                1+1
            elif ('/' in word):
                tmp.append('link_to_external_source')
            else:
                tmp.append(word)
        #print(tmp)
        reviews_ints.append(tmp)

    #print(reviews_ints)

    from collections import Counter
    from itertools import chain
    x = reviews_ints
    y = list(chain(*x))
    c = Counter(y)
    ## Build a dictionary that maps words to integers
    h = c.most_common()

    b = list()
    n = list()
    i = 1
    for label, num in h:
        b.append(label)
        n.append(i)
        i += 1

    vocab_to_int = OrderedDict(zip(b, n))
    vocab_to_int.update({'unkkown_word':len(n)})
    ## use the dict to tokenize each review in reviews_split
    ## store the tokenized reviews in reviews_ints

    data_ints = []
    #for review in reviews_ints:
        #data_ints.append([vocab_to_int[word] for word in review])

    for l in range(len(reviews_ints)):
        data_ints.append(reviews_ints[l])


    print(len(b))
    vocab_size = len(b)
##########################################TO Tensor then Pickle###########################################

    #print(text_data[1])
    #print(doc_clean[1])
    # print(lda_data[1])
    #print(X[1])


    import torch

    TRsplit = int(len(X)*0.75)
    CVsplit = int(len(X)*0.90)

    TY1 = Y1[CVsplit:]
    CY1 = Y1[TRsplit:CVsplit]
    Y1 = Y1[:TRsplit]

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
        tword_indexes.append(torch.LongTensor(tword_index[i]))

    cword_indexes = []
    for i in range(len(cword_index)):
        cword_indexes.append(torch.LongTensor(cword_index[i]))

    word_indexes = []
    for i in range(len(word_index)):
        word_indexes.append(torch.LongTensor(word_index[i]))

    tmeta_data = torch.Tensor(tmeta_data)
    cmeta_data = torch.Tensor(cmeta_data)
    meta_data = torch.Tensor(meta_data)

    TY = torch.Tensor(TY1)
    CY = torch.Tensor(CY1)
    Y = torch.Tensor(Y1)



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
    pickle_out = open("vocab_size.pickle", "wb")
    pickle.dump(vocab_size, pickle_out)
    pickle_out.close()

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

    pickle_out = open("vocab_to_int.pickle", "wb")
    pickle.dump(vocab_to_int, pickle_out)
    pickle_out.close()
main()
