import pickle


pickle_in = open('DOCLIST.pickle', 'rb')
DOCLIST = pickle.load(pickle_in)
pickle_in.close()

proc = []

#remove hashtags, ats, hhtps, rts an split
for text in DOCLIST:

    split = text.lower().split()
    ptext = []
    for word in split:
        if (
                #word.find(b'#') == -1 and
                word.find(b'@') == -1 and word.find(b'http') == -1 and word.find(b'rt') == -1 and word.find(b'&') == -1):
            ptext.append(word.decode('ascii'))
    proc.append(' '.join(ptext))

print(proc)


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

# Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

print("train")

# Running and Training LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=30, id2word = dictionary, alpha='auto', eval_every=5, passes = 50)

results = (ldamodel.show_topics(num_topics=10, num_words=5))

#print(results)

print(len(doc_term_matrix))

vectors = []
for i in range(0,len(doc_term_matrix)):
    #vectors.append(ldamodel[doc_term_matrix[i]])
    tmp = (ldamodel.get_document_topics(doc_term_matrix[i], minimum_probability=0.0))
    vector = []
    for scal in tmp:
        vector.append(scal[1])
    vectors.append(vector)

print(doc_clean[222])
print(doc_term_matrix[222])
print(DOCLIST[222])

pickle_out = open("LDA.pickle", "wb")
pickle.dump(vectors, pickle_out)
pickle_out.close()
