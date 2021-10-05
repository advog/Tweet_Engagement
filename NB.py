import pandas as pd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pickle

pickle_in = open('X.pickle', 'rb')
X = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('LDA.pickle', 'rb')
LDA = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('Y1.pickle', 'rb')
Y1 = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('Y2.pickle', 'rb')
Y2 = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('CX.pickle', 'rb')
CX = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('CY1.pickle', 'rb')
CY1 = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('CY2.pickle', 'rb')
CY2 = pickle.load(pickle_in)
pickle_in.close()

#for a in range(len(X)):
 #   X[a] = X[a][:25]
  #  for b in range(4,25):
   #     X[a][b]*=1

#print(X[1])

#print(Y1[1])

Y1 = np.array(Y1)
X = np.array(X)

print(X)


#results = sm.ZeroInflatedPoisson(Y1,X).fit()
results = sm.ZeroInflatedNegativeBinomialP(Y1,X).fit()
print(results.summary())

import math

mse = 0
count = 0
for i in range(len(CX)):
    Ypred = results.predict(CX[i])
    mse+= abs(Ypred[0] - CY1[i])
    count+=1

mse = mse/count
print(mse)





