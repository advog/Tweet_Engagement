import json
import pickle
import numpy as np




# read json data and return as a list of list
def readData(filename, tweetIDs):
    print(filename)
    toCSV = []
    with open(filename) as f:
        data = json.load(f)
        for tweet in data:
            tweetID = tweet['id']
            # check duplicate crawling tweet
            if tweetID in tweetIDs:
                continue
            tweetIDs[tweetID] = 1
            retweet_count = tweet['retweet_count']
            try:
                favorite_count = tweet['retweeted_status']['favorite_count']
            except:
                favorite_count = tweet['favorite_count']
            text = tweet['text'].encode('ascii', 'ignore')
            toCSV.append([tweetID, retweet_count, favorite_count, text])
    return toCSV, tweetIDs

def main():
    tweetIDs = {}
    fin = []
    for name in ['FDATobacco1.json', 'FDATobacco.json']:
        data, tweetIDs = readData(name, tweetIDs)
        fin = fin + data

    #######################################GET_FEATURES_GET_FEATURES_GET_FEATURES##########################################

    tweetList = fin + []


    allWords = []
    allHashtags = []
    allAted = []
    FINAL = []

    for tweets in tweetList:

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

        X.append(FINAL[num][4:8]+HASHVEC+ATVEC+WORDVEC)

        #X.append(FINAL[num][4:8])

        Y1.append(FINAL[num][1])
        Y2.append(FINAL[num][2])

        DOCLIST.append(FINAL[num][3])

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

    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel

    print("train")

    # Running and Training LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix[:3000], num_topics=20, id2word=dictionary, alpha='auto', eval_every=5, passes=50)

    for i in range(0, len(doc_term_matrix)):
        # vectors.append(ldamodel[doc_term_matrix[i]])
        tmp = (ldamodel.get_document_topics(doc_term_matrix[i], minimum_probability=0.0))
        vector = []
        for scal in tmp:
            vector.append(scal[1])
        X[i] = X[i]+vector

    ##########################################################AVG WORD EMBEDDINGS###################################################################

    # set the correct path to the file on your machine
    #model = gensim.models.KeyedVectors.load_word2vec_format('D:/EmbedStuff/glove.42B.300d.txt', binary=False)
    #""""
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
    #"""


    ##########################################################CV_SPLIT_CV_SPLIT_CV_SPLIT#############################################################





    CX = X[3000:]
    CY1 = Y1[3000:]
    CY2 = Y2[3000:]

    X = X[:3000]
    Y1 = Y1[:3000]
    Y2 = Y2[:3000]




    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("Y1.pickle", "wb")
    pickle.dump(Y1, pickle_out)
    pickle_out.close()

    pickle_out = open("Y2.pickle", "wb")
    pickle.dump(Y2, pickle_out)
    pickle_out.close()

    pickle_out = open("CX.pickle", "wb")
    pickle.dump(CX, pickle_out)
    pickle_out.close()

    pickle_out = open("CY1.pickle", "wb")
    pickle.dump(CY1, pickle_out)
    pickle_out.close()

    pickle_out = open("CY2.pickle", "wb")
    pickle.dump(CY2, pickle_out)
    pickle_out.close()

    pickle_out = open("clean_text.pickle", "wb")
    pickle.dump(doc_clean, pickle_out)
    pickle_out.close()

    #pickle_out = open("word_vectors.pickle", "wb")
    #pickle.dump(word_vectors, pickle_out)
    #pickle_out.close()

main()
