import json
import csv


# read json data and return as a list of list
def readData(filename, tweetIDs):
    print(filename)

    toCSV = []
    with open(filename) as f:
        data = []
        for line in f:
            temp = json.loads(line)
            data.append(temp)
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
            try:
                prefix_RT = tweet['full_text'].split(':')[0] + ': '
                text = prefix_RT + tweet['retweeted_status']['full_text'].encode('ascii', 'ignore')
            except:
                text = tweet['full_text'].encode('ascii', 'ignore')

            toCSV.append([tweetID, retweet_count, favorite_count, text])
    return toCSV, tweetIDs


import json
import pickle
import numpy as np


# read json data and return as a list of list
def readData2(filename, tweetIDs):
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

tweetIDs = {}
fin = []
for name in ['FDATobacco1.json', 'FDATobacco.json']:
     data, tweetIDs = readData2(name, tweetIDs)
     fin = fin + data

tweetIDs2 = {}
fin2 = []

filenames = ['FDATobacco3.json']
for name in filenames:
    data, tweetIDs = readData(name, tweetIDs)
    fin2 = fin2 + data

tweetlist = fin+fin2

import pickle

pickle_out = open("tweet_list_full.pickle", "wb")
pickle.dump(tweetlist, pickle_out)
pickle_out.close()

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = datapath('D:\\GLOVE\\glove.twitter.27B.200d.txt')
tmp_file = get_tmpfile("D:\\GLOVE\\twitter_word2vec.txt")

_ = glove2word2vec(glove_file, tmp_file)


print(len(tweetlist))