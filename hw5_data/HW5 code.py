# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 15:07:35 2021

@author: rgerr
"""
### IMPORTANT READ FIRST ###
# 1. Call setmainpath(path to hw5_data folder)
# 2. Call Naive()
###----------------------###
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error 
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import IncrementalPCA
from math import sqrt
import numpy as np

class Global:
    ### Temp for testing ###
    path1 = 'C:/Users/rgerr/OneDrive/Documents/MachineLearning/hw5_data/Heart/heart_trainSet.txt'
    path2 = 'C:/Users/rgerr/OneDrive/Documents/MachineLearning/hw5_data/Heart/heart_trainLabels.txt'
    path3 = 'C:/Users/rgerr/OneDrive/Documents/MachineLearning/hw5_data/Heart/heart_testSet.txt'
    ### actual paths ###
    heartTest = None
    heartTrainL = None
    heartTrainS = None
    groupsFolder = None
    train_y = None
    train_x = None
    test_x = None
    gtrain_x = None
    gtrain_y = None
    gtest_x = None
    gtest_y = None
    
#example for setmainpath -> 'C:/Users/rgerr/OneDrive/Documents/MachineLearning/hw5_data/'#
def setmainpath(folderpath):
    Global.groupsFolder = folderpath + '20groups/'
    Global.heartTest = folderpath + 'Heart/heart_testSet.txt'
    Global.heartTrainL = folderpath + 'Heart/heart_trainLabels.txt'
    Global.heartTrainS = folderpath + 'Heart/heart_trainSet.txt'
    Global.gtrain_x = folderpath + 'gisette/gisette_trainSet.txt'
    Global.gtrain_y = folderpath + 'gisette/gisette_trainLabels.txt'
    Global.gtest_x = folderpath + 'gisette/gisette_testSet.txt'
    Global.gtest_y = folderpath + 'gisette/gisette_testLabels.txt'
    

def readFile1(fileName):
        fileObj = open(fileName, "r") #opens the file in read mode
        file = fileObj.read().splitlines()
        array = []
        for line in file:
            tempa = []
            temp = line.split(',') #puts the file into an array
            for number in temp:
                tempa.append(float(number))
            array.append(tempa)
        fileObj.close()
        return array

def readFile2(filename):
    fileObj = open(filename,"r")
    array = fileObj.read().splitlines()
    fileObj.close()
    return array

def readFile3(filename):
    fileObj = open(filename,"r")
    array = fileObj.read().splitlines()
    temp = []
    for number in array:
        temp.append(float(number))
    array = np.array(temp)
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
                tempa.append(float(number))
            array.append(tempa)
        fileObj.close()
        return array

def readFile5(fileName):
        fileObj = open(fileName, "r") #opens the file in read mode
        file = fileObj.read().splitlines()
        array = {}
        for line in file:
            temp = line.split() #puts the file into an array
            array[temp[1]] = temp[0]
        fileObj.close()
        return array

#train_y = np.array(readFile2(Global.path2))
#train_x = np.array(readFile1(Global.path1))
#test_x = np.array(readFile1(Global.path3))

def Getprob1data():
    Global.train_y = np.array(readFile2(Global.heartTrainL))
    Global.train_x = np.array(readFile1(Global.heartTrainS))
    Global.test_x = np.array(readFile1(Global.heartTest))

# code for problem 1 #
    
def KNN(k,x,y,z):
    neigh = KNeighborsClassifier(n_neighbors=k,weights='distance')
    neigh.fit(x, y)
    pred = neigh.predict(z)
    return pred

#print(KNN(5,train_x,train_y,test_x))
    
def Leave1out():
    train_x, train_y = Global.train_x.astype(np.float64), Global.train_y.astype(np.float64)
    loo = LeaveOneOut()
    RMSE = []
    for train_index, test_index in loo.split(train_x):
        X_train, X_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        temp = []
        for k in range(10):
            k = k + 1
            model = KNeighborsClassifier(n_neighbors=k,weights='distance')
            model.fit(X_train, y_train)  #fit the model
            pred=model.predict(X_test) #make prediction on test set
            temp.append(sqrt(mean_squared_error(y_test,pred))) #calculate rmse
        RMSE.append(temp)
    npRMSE = np.array(RMSE)
    averageError = np.mean(npRMSE, axis=0)
    for i in range(10):
        print('Average RMSE value for all k= ' , i+1 , 'is:', averageError[i])
        
# code for Problem 2 #
        
def OrginizeData():
    X = readFile4(Global.groupsFolder+'test.data')
    Xt = readFile4(Global.groupsFolder+'train.data')
    X, Xt = helper(X), helper(Xt)
    return X, Xt
    
def helper(X):
    t = int(X[len(X)-1][0])
    #r = max(X, key=lambda x: x[1])
    ListOfDocs = [[0 for i in range(int(61188))] for j in range(t)]
    for L in X:
        i = int(L[0])
        j = int(L[1])
        ListOfDocs[i-1][j-1] = L[2]
        #print(int(L[0]))
        #while i > 0:
            #ListOfDocs[int(L[0])-1].append(float(L[1]))
            #i = i - 1
    return ListOfDocs

def Naive():
    X, Xt = OrginizeData()
    y = np.array(readFile2(Global.groupsFolder+'test.label'))
    yt = np.array(readFile2(Global.groupsFolder+'train.label'))
    clf = MultinomialNB()
    X, Xt = np.squeeze(np.array(X)), np.squeeze(np.array(Xt))
    clf.fit(Xt,yt)
    pred = clf.predict(X)
    MA = clf.score(X,y) # mean accuracy of the model
    TopicDict = readFile5(Global.groupsFolder+'test.map')
    f1 = open(Global.groupsFolder+'test_predict_Labels.txt',"w")
    f2 = open(Global.groupsFolder+'test_predict_Topic.txt',"w")
    f3 = open(Global.groupsFolder+'test_predict_Both.txt',"w")
    for line in pred:
        temp = str(line)+str(TopicDict[line])
        f1.write(str(line))
        f2.write(str(TopicDict[line]))
        f3.write(temp)
    f1.close()
    f2.close()
    f3.close()
    return MA


# code for Problem 3 #
def getDatafor3alt():
    train_y = np.array(readFile3(Global.gtrain_y))
    train_x = np.array(readFile4(Global.gtrain_x))
    test_x = np.array(readFile4(Global.gtest_x))
    test_y = np.array(readFile3(Global.gtest_y))
    return train_x, train_y, test_x, test_y

# code for Problem 3 #
def getDatafor3():
    train_y = np.array(readFile3(Global.gtrain_y))
    train_x = np.array(readFile4(Global.gtrain_x))
    X, y = train_x, train_y
    return X, y
    
def leave1out2(P):
    gtrain_x, gtrain_y = getDatafor3()
    if P == True:
        pca = IncrementalPCA()
        pca.fit(gtrain_x)
        gtrain_x = pca.transform(gtrain_x)
    loo = LeaveOneOut()
    RMSE = []
    for train_index, test_index in loo.split(gtrain_x):
        X_train, X_test = gtrain_x[train_index], gtrain_x[test_index]
        y_train, y_test = gtrain_y[train_index], gtrain_y[test_index]
        temp = []
        for k in range(10):
            k = k + 1
            model = KNeighborsClassifier(n_neighbors=k,weights='distance')
            model.fit(X_train, y_train)  #fit the model
            pred=model.predict(X_test) #make prediction on test set
            temp.append(sqrt(mean_squared_error(y_test,pred))) #calculate rmse
        RMSE.append(temp)
    npRMSE = np.array(RMSE)
    averageError = np.mean(npRMSE, axis=0)
    for i in range(10):
        print('Average RMSE value for all k= ' , i+1 , 'is:', averageError[i])

def KNN3(k):
    x, y, z, v = getDatafor3alt()
    neigh = KNeighborsClassifier(n_neighbors=k,weights='distance')
    neigh.fit(x, y)
    scor = neigh.score(z,v)
    print('The Accuracy of the model on testing data is', scor, 'for k= ', k)

def KNNpca(k):
    x, y, z, v = getDatafor3alt()
    pca = IncrementalPCA()
    pca.fit(x)
    x = pca.transform(x)
    #pca.fit(z)
    #z = pca.transform(z)
    neigh = KNeighborsClassifier(n_neighbors=k,weights='distance')
    neigh.fit(x, y)
    scor = neigh.score(z,v)
    print('The Accuracy of the model on testing data is', scor, 'for k= ', k)


def lens(X):
    for L in X:
        print(len(L))

#    for key in X:
 #       y = []
  #      temp = X[key]
   #     for j in temp:
    #        i = j[1]
     #       while i > 0:
      #          y.append(float(j[0]))
       #         i=i-1
        #tempX.append(list(y))
#    for k in Xt:
 #       r = []
  #      tempa = Xt[k]
   #     for t in tempa:
    #        a = t[1]
     #       while a > 0:
      #          r.append(float(t[0]))
       #         a=a-1
        #tempXt.append(list(r))