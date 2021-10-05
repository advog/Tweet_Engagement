import json
import sys
import csv
import numpy as np
import pickle


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

    # since FDATobacco1 crawled more recently than FDATobacco, some counts make updated,
    # that's why I parsed FDATobacco1 first

    filenames = ['FDATobacco1.json', 'FDATobacco.json']
    for name in filenames:
        data, tweetIDs = readData(name, tweetIDs)
        fin = fin + data

    allWords = []
    allHashtags = []
    allAted = []
    FINAL = []

    for tweets in fin:

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

    import random
    random.shuffle(FINAL)

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

        #X.append(FINAL[num][4:8]+HASHVEC+ATVEC+WORDVEC)

        X.append(FINAL[num][4:8])

        Y1.append(FINAL[num][1])
        Y2.append(FINAL[num][2])

        DOCLIST.append(FINAL[num][3])

    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("Y1.pickle", "wb")
    pickle.dump(Y1, pickle_out)
    pickle_out.close()

    pickle_out = open("Y2.pickle", "wb")
    pickle.dump(Y2, pickle_out)
    pickle_out.close()

    pickle_out = open("DOCLIST.pickle", "wb")
    pickle.dump(DOCLIST, pickle_out)
    pickle_out.close()

main()
