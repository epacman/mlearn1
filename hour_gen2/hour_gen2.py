# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np

import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#from mpl_finance import candlestick2_ohlc


what_hour = []
oopen = []
close = []
high = []
low = []
high_temp = 0
low_temp = 2000

for i in range(1,len(minute)):
    
    high_temp = max(kurs[i],high_temp)
    low_temp = min(kurs[i],low_temp)
    if minute[i] < minute[i-1]:
        #hour has changed
        oopen.append(kurs[i])
        close.append(kurs[i-1]) #egentligen förra timmens close
        high.append(high_temp)
        low.append(low_temp)
        what_hour.append(hour[i])
        high_temp = 0
        low_temp = 2000
        
high = high[1::]
low = low[1::]
close = close[1::]
        
#what_hour släpar efter en timme
        
candles = []
temp = []

for j in range(4,len(oopen)-1): 
    norm = np.mean([oopen[j-2],close[j-2]])
    
    temp = [what_hour[j]-1, close[j-4]/norm,\
            oopen[j-3]/norm, high[j-3]/norm,low[j-3]/norm, close[j-3]/norm, \
            oopen[j-2]/norm, high[j-2]/norm,low[j-2]/norm, close[j-2]/norm, \
            oopen[j-1]/norm, high[j-1]/norm,low[j-1]/norm, close[j-1]/norm, \
            oopen[j]/norm, high[j]/norm,low[j]/norm, close[j]/norm,\
            close[j] > close[j-1]]
    
    candles.append(temp)
    
Y = []
X = []
    
for k in range(len(candles)):
    Y.append(candles[k][18])
    X.append(candles[k][1:13])
    
    
validation_size = 0.2
seed = 154
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed, shuffle=True)

knn = LogisticRegression()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))