#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 09:52:28 2018

@author: Cassidy
"""

import sys
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
dataSet = namedtuple('trainingSet','X Y')

def reduceData(path):
    inputData = namedtuple('data','Label Series')
    dataDict = {}
    exoPlanets = []
    
    with open(path, 'r') as g:
        variableNames = g.readline()
        #print(variableNames.split(',')[:10])
        for k, record in enumerate(g):
            data = record.strip('\n').split(',')
            dataDict[k] = inputData(int(data[0]), [float(x) for x in data[1:]] )
            if int(data[0]) is 2: 
                exoPlanets.append(k)
        
    labelTable =[0]*2
    for value in dataDict.values():        
        labelTable[value.Label==2] += 1
    print('N exoplanet stars = ',labelTable[1], 'N stars w/out exoplanets = ',
          labelTable[0])    
    
    inc = .20
    pVec = [p for p in np.arange(.5*inc, 1 - .5*inc, inc)]

    N = len(dataDict)
    dataSet = namedtuple('data','X Y')
    
    Y = np.matrix(np.zeros( shape = (N, 1) ) )
    X = np.matrix(np.ones(shape = (N, len(pVec) + 1)))
    for key, (group, y) in dataDict.items():
        
        Y[key] = group - 1
        xp = np.percentile(y,[25,75])
        trY = [val for val in y if  xp[0] < val < xp[1] ]
        y = np.percentile( (y - np.mean(trY)) / np.std(trY), pVec)        
        X[key,1:] = y
    D = dataSet(X, Y)    
    return D

def dim(X):
    return np.shape(X)
        
path = '/Users/Cassidy/Documents/Assignments/S2018/M462/Data/exoTrain.csv'
D = reduceData(path)

X = D.X
Y = D.Y
n, q = dim(D.X)


########################
'''Logistic Regression'''

def objFn(X, Y, b):
    return -sum(Y.T*X*b)+sum(np.log(1+np.exp(X*b)))

#Create initial estimate of b as numpy matrix
b = np.matrix(np.zeros(shape = (q, 1)))
b = np.linalg.solve(X.T*X,X.T*Y)
piVector = 1/(1+np.exp(-X*b))

#Create loop that iterates until change between successive
#evaluations is less than some e
previousValue = 1

objFnValue = 0

WX = X.copy()

abs(previousValue - objFnValue)

k=0
while (abs(previousValue - objFnValue)) > 1e-4:
    previousValue = objFnValue
    WX = X.copy()
    for i, x in enumerate(WX):
        WX[i,:] *= float((1 - piVector[i])*piVector[i])
    b += np.linalg.solve(X.T*WX, X.T*(Y - piVector))
    piVector = 1/(1+np.exp(-X*b))
    objFnValue = objFn(X, Y, b)
    print(k, objFnValue)
    k += 1
    
print(b)

#Compute Accuracy
appAcc = 0
for i in range (n):
    prediction=int(piVector[i]>.5)
    appAcc += (prediction==float(Y[i])) 
appAcc/n

#########
prob = piVector

##############

#List containing actual labels
act = [int(y[0])for y in np.matrix.tolist(Y)]
#List containing estimated probabilities
piVector

#############

thresholds = np.arange(.001,.0038,.001)
'''confusion matrix'''
cm = np.zeros(shape = (2, 2))
sensitivity = [0]*len(thresholds)
specificity = [0]*len(thresholds)

############
for k,p in enumerate(thresholds):
    cm = np.zeros(shape = (2, 2))
    for i in range(n):
        pred = int(piVector[i] >= p)
        cm[act[i],pred] += 1
    n1 = (cm[1,0] + cm[1,1])
    n2 = (cm[0,0] + cm[0,1])
    sensitivity[k] = (cm[1,1])/n1
    specificity[k] = (cm[0,0])/n2
    
print(cm)
sensitivity[k]
specificity[k]

####Sensitivity###
n1 = (cm[1,0] + cm[1,1])
n2 = (cm[0,0] + cm[0,1])
cm[1,1]

sensitivity[k] = (cm[1,1])/n1
specificity[k] = (cm[0,0])/n2

import matplotlib.pyplot as plt
plt.plot(thresholds, sensitivity)
plt.plot(thresholds, specificity)
plt.xlabel('Threshold', fontsize=15)
plt.ylabel('Sensitivity/Specificity', fontsize=15)




'''End of star program'''


    
#########################
'''Tutorial 1 – Federalist papers'''

import numpy as np
from collections import Counter
import re
import operator
import nltk
from nltk.corpus import stopwords
from collections import namedtuple
nltk.download('stopwords')
stopwords = stopwords.words('english')


logPriors = dict.fromkeys(authors,0)
freqDistnDict = dict.fromkeys(authors)
for label in trainLabels:
    number, author = label
    D = wordDict[label]
    distn = freqDistnDict.get(author)
    if distn is None:
        distn = D
    else:
        for word in D:
            value = distn.get(word)
            if value is not None:
                distn[word] += D[word]
            else:
                distn[word] = D[word]
    freqDistnDict[author] = distn
    logPriors[author] +=1
    
disputedList = [18,19,20,49,50,51,52,53,54,55,56,57,58,62,63]
for label in wordDict:
    number, author = label
    if number not in disputedList:
        trainLabels = []
        trainLabels.append(label)
print(len(trainLabels))     
    
nR = len(trainLabels)
logProbDict = dict.fromkeys(authors,{})
distnDict = dict.fromkeys(authors)
for author in authors:
    authorDict = {}
    logPriors[author] = np.log(logPriors[author]/nR)
    nWords = sum([freqDistnDict[author][word] for word in commonList ])
    print(nWords)
    for word in commonList:
        relFreq = freqDistnDict[author][word]/nWords
        authorDict[word] = np.log(relFreq)
    distnDict[author]  = [logPriors[author], authorDict]

stopWordSet = set(stopwords) | set(' ') 
print(len(stopWordSet))        

identifier = namedtuple('label','index author')
authors = ['JAY', 'MADISON','HAMILTON' ]

paperDict = {}
with open(path, "rU") as f: 
    for string in f:
        string = string.replace('\n',' ')
        print(string)
        key, value = string.split(',')
        paperDict[int(key)] = value.replace(' ','')
        
paperDict = {}
   path = ’../Data/owners.txt’
   with open(path, "rU") as f:
       for string in f:
           string = string.replace(’\n’, ’ ’)
           print(string)

path = '/Users/Cassidy/Documents/Assignments/S2018/M462/Data/1404-8.txt'
sentenceDict = {}
nSentences = 0
sentence = ''
String = ''
with open(path, "rU") as f: 
    for string in f:
        string = string.replace('\n',' ')
        String = String+string
        
############################### 
positionDict = {}
opening  = 'To the People of the State of New York'
counter = 0
for m in re.finditer(opening, String):
    counter+= 1
    positionDict[counter] = [m.end()]

close  = 'PUBLIUS'
counter = 0
for m in re.finditer(close, String):
    counter+= 1
    positionDict[counter].append(m.start())

wordDict = {}

paperCount = 0          
for paperNumber in positionDict:
    b,e = positionDict[paperNumber]
    author = paperDict[paperNumber]
    label = identifier(paperNumber,author)
    paper = String[b+1:e-1]

    for char in '.,?!/;:-"()':
        paper = paper.replace(char,'')
    paper = paper.lower().split(' ')
    for sw in stopWordSet:
        paper = [w for w in paper if w != sw]
        
    #freqDict = Counter(paper)
        
    wordDict[label] = Counter(paper)
    print(label.index,label.author,len(wordDict[label]))
        

table = dict.fromkeys(set(paperDict.values()),0)
for label in wordDict:
    table[label.author] += 1
print(table)    

############    

####################### 

usedDict = {}
for label in trainLabels:
    print(label.author,len(wordDict[label]))
    words = list(wordDict[label].keys())
    for word in words:
        value = usedDict.get(word)
        if value is None:
            usedDict[word] = set([label.author])
        else:
            usedDict[word] = value | set([label.author])

commonList = [word for word in usedDict if len(usedDict[word] ) == 3] 
len(commonList)

#################################### shortening wordList to only common
for label in wordDict:
    D = wordDict[label]
    newDict = {}
    for word in D:
        if word in commonList:
            newDict[word] = D[word]
    wordDict[label] = newDict        
    print(label,len(wordDict[label])) 
    
################
logPriors = dict.fromkeys(authors,0)
freqDistnDict = dict.fromkeys(authors)
for label in trainLabels:
    number, author = label
    D = wordDict[label]
    distn = freqDistnDict.get(author)
    if distn is None:
        distn = D
    else:
        for word in D:
            value = distn.get(word)
            if value is not None:
                distn[word] += D[word]
            else:
                distn[word] = D[word]
    freqDistnDict[author] = distn
    logPriors[author] +=1
    
    
##################
nR = len(trainLabels)
logProbDict = dict.fromkeys(authors,{})
distnDict = dict.fromkeys(authors)
for author in authors:
    authorDict = {}
    logPriors[author] = np.log(logPriors[author]/nR)
    nWords = sum([freqDistnDict[author][word] for word in commonList ])
    print(nWords)
    for word in commonList:
        relFreq = freqDistnDict[author][word]/nWords
        authorDict[word] = np.log(relFreq)
    distnDict[author]  = [logPriors[author], authorDict]
    
##################
nGroups = len(authors)
confusionMatrix = np.zeros(shape = (nGroups,nGroups))
skip = [18,19,20,49,50,51,52,53,54,55,56,57,58,62,63]
for label in wordDict:
    testNumber, testAuthor = label
    if testNumber not in skip:
        xi = wordDict[label]
        postProb = dict.fromkeys(authors,0)
        for author in authors:
            logPrior, logProbDict = distnDict[author]
            postProb[author] = logPrior + sum([xi[word]*logProbDict[word] for word in xi])
        postProbList = list(postProb.values())
        postProbAuthors = list(postProb.keys())
        maxIndex = np.argmax(postProbList)
        prediction = postProbAuthors[maxIndex]
        print(testAuthor,prediction)
        i = list(authors).index(testAuthor)
        j = list(authors).index(prediction)
        confusionMatrix[i,j] += 1
        
print(confusionMatrix)
print(’acc = ’,sum(np.diag(confusionMatrix))/sum(sum(confusionMatrix)))





####################
import re
import operator

path = '/Users/Cassidy/Documents/Assignments/S2018/M462/Data/stopWord.txt'

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords = stopwords.words('english')
stopWordSet = set(stopwords) | set(' ') 
print(len(stopWordSet)) 

def stopWordDict() :
    ''' use this if there are problems with the stop_words module '''
    stopWordList = [] # from http://www.ranks.nl/stopwords
    path = '/Users/Cassidy/Documents/Assignments/S2018/M462/Data/stopWord.txt'
    with open(path, "rU") as f:
        for word in f:
            stopWordList.append(word.strip('\n'))
    stopWordSet = set(stopWordList)
    print(stopWordSet)
    return stopWordSet            
stopWordSet=stopWordDict()

print(len(stopWordSet))        

from collections import namedtuple
identifier = namedtuple('label','index author')
authors = ['JAY', 'MADISON', 'HAMILTON']

import numpy as np
from collections import Counter

path = '/Users/Cassidy/Documents/Assignments/S2018/M462/Data/owners.txt'
paperDict = {}
with open(path, "rU") as f: 
    for string in f:
        string = string.replace('\n', ' ')
        print(string)
        key, value = string.split(',')
        paperDict[int(key)] = value.replace(' ','')


path = '/Users/Cassidy/Documents/Assignments/S2018/M462/Data/1404-8.txt'
sentenceDict = {}
nSentences = 0
sentence = ''
String = ''
with open(path, "rU") as f: 
    for string in f:
        string = string.replace('\n',' ')
        String = String+string
        
    
        
############################### 
positionDict = {}
opening  = 'To the People of the State of New York'
counter = 0
for m in re.finditer(opening, String):
    counter+= 1
    positionDict[counter] = [m.end()]

close  = 'PUBLIUS'
counter = 0
for m in re.finditer(close, String):
    counter+= 1
    positionDict[counter].append(m.start())

wordDict = {}
for paperNumber in positionDict:
    b, e = positionDict[paperNumber]
    author = paperDict[paperNumber]
    label = (paperNumber,author)
    paper = String[b+1:e-1]
    for char in '.,?!/;:-"()':
        paper = ''.join(paper)
        paper = paper.replace(char,'')
        paper = paper.lower().split(',')
    for sw in stopWordSet:
        paper = [w for w in paper if w != sw]
        
        wordDict[label] = Counter(paper)
        print(label.index,label.author,len(wordDict[label]))
        
table = dict.fromkeys(set(paperDict.values()),0)
for label in wordDict:
    table[label.author] += 1
print(table)    

############ 

disputedList = [18,19,20,49,50,51,52,53,54,55,56,57,58,62,63]
trainLabels = []
for label in wordDict:
    number, author = label
    if number not in disputedList:
        trainLabels.append(label)
print(len(trainLabels))    

####################### 

usedDict = {}
for label in trainLabels:
    print(label.author,len(wordDict[label]))
    words = list(wordDict[label].keys())
    for word in words:
        value = usedDict.get(word)
        if value is None:
            usedDict[word] = set([label.author])
        else:
            usedDict[word] = value | set([label.author])

commonList = [word for word in usedDict if len(usedDict[word] ) == 3] 
len(commonList)

#################################### shortening wordList to only common
for label in wordDict:
    D = wordDict[label]
    newDict = {}
    for word in D:
        if word in commonList:
            newDict[word] = D[word]
    wordDict[label] = newDict        
    print(label,len(wordDict[label]))
    
    
##########################
logPriors = dict.fromkeys(authors,0)
freqDistnDict = dict.fromkeys(authors)
for label in trainLabels:
    number, author = label
    D = wordDict[label]
    distn = freqDistnDict.get(author)
    if distn is None:
        distn = D
    else:
        for word in D:
            value = distn.get(word)
            if value is not None:
                distn[word] += D[word]
            else:
                distn[word] = D[word]
    freqDistnDict[author] = distn
    logPriors[author] +=1
    
nR = len(trainLabels)
   logProbDict = dict.fromkeys(authors,{})
   distnDict = dict.fromkeys(authors)
   for author in authors:
       authorDict = {}
       logPriors[author] = np.log(logPriors[author]/nR)
       nWords = sum([freqDistnDict[author][word] for word in commonList ])
       print(nWords)
       for word in commonList:
           relFreq = freqDistnDict[author][word]/nWords
           authorDict[word] = np.log(relFreq)
       distnDict[author]  = [logPriors[author], authorDict]
       
       
nGroups = len(authors)
confusionMatrix = np.zeros(shape = (nGroups,nGroups))
skip = [18,19,20,49,50,51,52,53,54,55,56,57,58,62,63]
for label in wordDict:
    testNumber, testAuthor = label
    if testNumber not in skip:
        xi = wordDict[label]
        postProb = dict.fromkeys(authors,0)
        for author in authors:
            logPrior, logProbDict = distnDict[author]
            postProb[author] = logPrior
                + sum([xi[word]*logProbDict[word] for word in xi])
        postProbList = list(postProb.values())
        postProbAuthors = list(postProb.keys())
        maxIndex = np.argmax(postProbList)
        prediction = postProbAuthors[maxIndex]
        print(testAuthor,prediction)
        i = list(authors).index(testAuthor)
        j = list(authors).index(prediction)
        confusionMatrix[i,j] += 1
print(confusionMatrix)
print(’acc = ’,sum(np.diag(confusionMatrix))/sum(sum(confusionMatrix)))
