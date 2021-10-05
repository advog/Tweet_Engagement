import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.linear_model
from sklearn.svm import SVR
from sklearn import metrics
import pickle

pickle_in = open('Y.pickle', 'rb')
Y = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open('X.pickle', 'rb')
X = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open('names.pickle', 'rb')
names = pickle.load(pickle_in)
pickle_in.close()

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=0)

thingys = [sklearn.linear_model.LogisticRegression(max_iter = 1000),
    ]

for thing in thingys:
    regression = thing
    regression.fit(X_train,y_train)
    y_pred=regression.predict(X_test)
    y_pred = y_pred.tolist()

    print(regression.__class__.__name__)
    print(y_pred)
    print(y_test)
    print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
    print('macro f1: ',metrics.f1_score(y_test, y_pred,average='macro'))
    print('weighted f1: ',metrics.f1_score(y_test, y_pred,average='weighted'))
    print('micro f1: ',metrics.f1_score(y_test, y_pred,average='micro'))

    importance = regression.coef_
    #summarize feature importance
    th = 0.8

    nam = ['hash','rt','at','image'] + ['lda']*20 + names


for w in range(5):
    print('Ordinal class: ' + str(w))
    for i,v in enumerate(importance[w]):

        if(abs(v) > th or True):

            print('Feature:%0d:Score:%.5f' % (i,v) +':Name:' + nam[i])



print('report: ',metrics.classification_report(y_test, y_pred))