# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 12:16:16 2018

@author: elindgre

#help(“FunctionName”)
"""

 #Python version
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


# Load dataset
url = "mlearndata_clean_procent_aprildec18_alladata.csv"
#names = ['first_minute', 'lunch', 'last_minute', 'yclose', 'yopen','ma5','ma5procent','ytot', 'closetopopen', 'opentolunch', 'lunchtoclose','rek']

names = ['first_minute', 'lunch', 'last_minute', 'yclose', 'yopen','ytot', 'closetopopen', 'opentolunch', 'lunchtoclose','rek']

dataset = pandas.read_csv(url, names=names)

#print dataset.shape
#print(dataset.head(20))
#print(dataset.describe())
#print(dataset.groupby('class').size())


# box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()
#
#dataset.hist()
#plt.show()
#
#scatter_matrix(dataset)
#plt.show()
start = 0
stop = -1
#960

# använd 20% som validation
#vill använda kolumner 0,1,3 
array = dataset.values
X = array[start:stop,3:5]
Y = array[start:stop,6]
opening = array[:,0]

removed = 0
for i in range(len(X)):
    if X[i-removed,0] < -1.1 or  abs(X[i-removed,1]) > 5:
        X = np.delete(X,i-removed, axis=0)
        Y = np.delete(Y,i-removed)
        removed += 1


X = np.insert(X,2, 0,axis=1)
X = np.insert(X,3, 0,axis=1)
X = np.insert(X,4, 0,axis=1)
period = 0
for i in range(len(X)):
    if i >= 2:
        X[i,2] = X[i-2,0] #gårdagens close-open
        X[i,3] = X[i-2,1]   #gårdagens  open-lunch
        X[i,4] = array[i-2,5] #gårdagens lunch-close
        

        
        

#Ynum = array[:,8] #Antal punkter efter lunch
    
#for i in range(len(Y)):
#    Y[i] = int(Y[i])
#Y=Y.astype('int')
    
validation_size = 0.15
seed = 18
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed, shuffle=True)

scoring = 'accuracy'


#%%


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

#%%
#fig = plt.figure()
#fig.suptitle('Algorithm Comparison')
#ax = fig.add_subplot(111)
#plt.boxplot(results)
#ax.set_xticklabels(names)
#plt.show()
#
# 
#

# Make predictions on validation dataset
#knn = KNeighborsClassifier()
knn = GaussianNB()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

#%%


close = 1396
openk = 1397
lunch = 1420
##
##
a= (openk - close)/openk * 100
b= (lunch - openk)/openk * 100
#c = 
#   
#print(knn.predict([[a,b]]))
#
##Rör sig i snitt 7 punkter efter lunch. Index 1425 i snitt för dataset, 
##så 0,5 % efter lunch i snitt men i spann 0,43-0,58% beroende på när i tid
#print(np.mean(np.abs(Ynum)))
#print(np.mean(openk))
#
##Kolla procentuellt också, inte bara antal punkter
#






