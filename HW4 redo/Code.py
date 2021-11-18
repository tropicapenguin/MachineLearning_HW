# -*- coding: utf-8 -*-
"""
Created on Tue May 11 18:44:56 2021

@author: rgerr
"""


import numpy as np
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.model_selection import train_test_split

class g:
    folder = 'F:/Desktop/HW4 redo/'
    Etest = 'E2006.test.txt'
    Etrain = 'E2006.train.txt'
    house = 'house.txt'
    house_scale = 'house_scale.txt'

def setmainpath(folderpath):
    g.folder = folderpath

def readFiles(filename):
    data = np.load(filename)
    return data

def readFilesP(filename):
    data = np.load(filename, allow_pickle=True)
    return data

def readFiles1(filename):
    fileObj = open(filename,"r")
    array = fileObj.read().splitlines()
    fileObj.close()
    return array

def readFile4(fileName):
        fileObj = open(fileName, "r") #opens the file in read mode
        file = fileObj.read().splitlines()
        array = []
        for line in file:
            tempa = []
            temp = line.split() #puts the file into an array
            for number in temp:
                tempa.append(number)
            array.append(tempa)
        fileObj.close()
        return array
    
def getresults(array):
    temp = array
    results = []
    for i in temp:
        x = i.pop(0)
        results.append(float(x))
    return results, temp
    
def numberfy(x):
    tempa = []
    for i in x:
        temp = []
        k = 1
        for j in i:
            tempL = []
            tempL = j.split(':')
            temp.append(float(tempL[1]))
            k = k + 1
            tempa.append(temp)
    return tempa

def ridged():
    data = readFile4(g.folder+g.house_scale)
    y, x = getresults(data)
    x = numberfy(x)
    alpha = [0.01,0.1,1.0,10.0,100.0]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
    extra = 2 * len(y_train)
    results = []
    for i in alpha:
        clf = Ridge(alpha=(i*extra), normalize=True)
        clf.fit(x_train,y_train)
        j = clf.score(x_test, y_test)
        results.append([j, alpha])
        

def ridgedpt2():
    data = readFile4(g.folder+g.house_scale)
    y, x = getresults(data)
    x = numberfy(x)
    alpha = [0.01,0.1,1.0,10.0,100.0]
    x_train, x_test, y_train, y_test = x[0:399], x[400:], y[0:399], y[400:]
    extra = 2 * len(y_train)
    results = []
    for i in alpha:
        clf = Ridge(alpha=(i*extra), normalize=True)
        clf.fit(x_train,y_train)
        j = clf.score(x_test, y_test)
        results.append([j, alpha])

