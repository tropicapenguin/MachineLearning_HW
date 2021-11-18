# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:14:59 2021

@author: rgerr
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error 
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

class top:
    ### main path for data ###
    breast = "b/breast-cancer_scale.txt"
    sonar = "s/sonar_scale.txt"
    covtype = "c/covtype.data"
    ### index paths for testing ###
    breast_testing = "b/breast-cancer-scale-test-indices.txt"
    sonar_testing = "s/sonar-scale-test-indices.txt"
    covtype_testing = "c/covtype.test.index.txt"
    ### index paths for training ###
    breast_training = "b/breast-cancer-scale-train-indices.txt"
    sonar_training = "s/sonar-scale-train-indices.txt"
    covtype_training = "c/covtype.train.index.txt"
    ### folder path ###
    groupsFolder = "C:/Users/rgerr/Documents/MachineLearning/HW6/"
    ### data sets ###
    
    ### error messages ###
    error = "You entered the wrong letter, Only c, b, and s are options"
    
#example for setmainpath -> 'C:/Users/rgerr/Documents/MachineLearning/'#
# or 'C:/Users/rgerr/OneDrive/Documents/' #
# first call setmainpath(folderpath) to be able to run any other fucntion calls #
def setmainpath(folderpath):
    top.groupsFolder = folderpath + 'HW6/'
    
    ### misc read in function ###
def readFiles(filename):
    fileObj = open(filename,"r")
    array = fileObj.read().splitlines()
    fileObj.close()
    return array

    ### for data ###
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
    
def readFileC(fileName):
        fileObj = open(fileName, "r") #opens the file in read mode
        file = fileObj.read().splitlines()
        array = []
        for line in file:
            tempa = []
            temp = line.split(',') #puts the file into an array
            for number in temp:
                tempa.append(number)
            array.append(tempa)
        fileObj.close()
        return array
    
    ### getting the results of / identies of each line ###
def getresults(array):
    temp = array
    results = []
    for i in temp:
        x = i.pop(0)
        results.append(float(x))
    return results, temp
        
def numberfy(x, L):
    tempa = []
    for i in x:
        temp = []
        k = 0
        for j in i:
            if k > 8:
                t = j[3:]
            else:
                t = j[2:]
            temp.append(float(t))
            k = k + 1
        if L == 's':
            if len(temp) == 60:
                tempa.append(temp)
            else:
                temp.append(0)
                tempa.append(temp)
        else:
            tempa.append(temp)
    return tempa

def covtyperesult(x):
    temp = x
    results = []
    for i in temp:
        x = i.pop(len(i)-1)
        results.append(float(x))
    return results, temp

def numberfyC(x):
    tempa = []
    for i in x:
        temp = []
        for j in i:
            temp.append(float(j))
        tempa.append(temp)
    return tempa
    ### end of data formating for data ###
    
    ### getting the data ###
def TnT(letter):
    if letter == 'b':
        testing = readFiles(top.groupsFolder+top.breast_testing)
        training = readFiles(top.groupsFolder+top.breast_training)
        data = readFile4(top.groupsFolder+top.breast)
    elif letter == 's':
        testing = readFiles(top.groupsFolder+top.sonar_testing)
        training = readFiles(top.groupsFolder+top.sonar_training)
        data = readFile4(top.groupsFolder+top.sonar)
    elif letter == 'c':
        testing = readFiles(top.groupsFolder+top.covtype_testing)
        training = readFiles(top.groupsFolder+top.covtype_training)
        data = readFileC(top.groupsFolder+top.covtype)
    else:
        return top.error
    if letter == 'b' or letter == 's':
        y, x = getresults(data)
        x = numberfy(x, letter)
    else:
        y, x = covtyperesult(data)
        x = numberfyC(x)
    te = []
    tr = []
    tractual = []
    teactual = []
    for i in testing:
        teactual.append(y[int(i)-1])
        te.append(x[int(i)-1])
    for j in training:
        tractual.append(y[int(j)-1])
        tr.append(x[int(j)-1])
    return te, tr, tractual, teactual

def fold(letter):
    if letter == 'b':
        data = readFile4(top.groupsFolder+top.breast)
    elif letter == 's':
        data = readFile4(top.groupsFolder+top.sonar)
    elif letter == 'c':
        data = readFileC(top.groupsFolder+top.covtype)
    else:
        return top.error
    kf = KFold(n_splits=5)
    if letter == 'b' or letter == 's':
        y, X = getresults(data)
        X = numberfy(X, letter)
    else:
        y, X = covtyperesult(data)
        X = numberfyC(X)
    for train_index, test_index in kf.split(X):
        X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
    return X_test, X_train, y_train, y_test
    
    ### Machine Learning applications ###
# options for letter input = 's','b','c'
# options for f input = 'y','n'
# options for prep = 's','n'
# these are the only input options you can plug in #

def regress(letter, f, prep):
    if f == "y":
        TestingX, TrainingX, Trainingy, Testingy = fold(letter)
    elif f == "n":
        TestingX, TrainingX, Trainingy, Testingy = TnT(letter)
    else:
        return "fold can only be either y or n"
    Cs = [0.1,1.0,10.0,100.0,1000.0]
    final = []
    if prep == 's':
        scaler = preprocessing.StandardScaler().fit(TrainingX)
        TrainingX = scaler.transform(TrainingX)
        TestingX = scaler.transform(TestingX)
    elif prep == 'n':
        normalizer = preprocessing.Normalizer().fit(TrainingX)
        TrainingX = normalizer.transform(TrainingX)
        TestingX = normalizer.transform(TestingX)
    for i in Cs:
        f = make_pipeline(StandardScaler(), LogisticRegression(C = i, max_iter = 2000))
        f.fit(TrainingX, Trainingy)
        MSE = mean_squared_error(Testingy, f.predict(TestingX))
        result = f.score(TestingX, Testingy)
        final.append([result,MSE,i])
    return final
        
# this one is used for the Covtype data set in problem 3 #
# otherwise this is also used in problem 2 #
def SVML(letter, f, prep):
    if f == "y":
        TestingX, TrainingX, Trainingy, Testingy = fold(letter)
    elif f == "n":
        TestingX, TrainingX, Trainingy, Testingy = TnT(letter)
    else:
        return "fold can only be either y or n"
    Cs = [0.1,1.0,10.0,100.0,1000.0]
    final = []
    if prep == 's':
        scaler = preprocessing.StandardScaler().fit(TrainingX)
        TrainingX = scaler.transform(TrainingX)
        TestingX = scaler.transform(TestingX)
    elif prep == 'n':
        normalizer = preprocessing.Normalizer().fit(TrainingX)
        TrainingX = normalizer.transform(TrainingX)
        TestingX = normalizer.transform(TestingX)
    for i in Cs:
        fi = make_pipeline(StandardScaler(), LinearSVC(C = i, dual = False, max_iter = 2000))
        fi.fit(TrainingX, Trainingy)
        MSE = mean_squared_error(Testingy, fi.predict(TestingX))
        result = fi.score(TestingX, Testingy)
        final.append([result,MSE,i])
    return final

def SVMR(letter, f, prep):
    if f == "y":
        TestingX, TrainingX, Trainingy, Testingy = fold(letter)
    elif f == "n":
        TestingX, TrainingX, Trainingy, Testingy = TnT(letter)
    else:
        return "fold can only be either y or n"
    Cs = [0.1,1.0,10.0,100.0,1000.0]
    final = []
    if prep == 's':
        scaler = preprocessing.StandardScaler().fit(TrainingX)
        TrainingX = scaler.transform(TrainingX)
        TestingX = scaler.transform(TestingX)
    elif prep == 'n':
        normalizer = preprocessing.Normalizer().fit(TrainingX)
        TrainingX = normalizer.transform(TrainingX)
        TestingX = normalizer.transform(TestingX)
    for i in Cs:
        fi = make_pipeline(StandardScaler(), SVC(C = i, kernel='rbf', gamma='auto'))
        fi.fit(TrainingX, Trainingy)
        MSE = mean_squared_error(Testingy, fi.predict(TestingX))
        result = fi.score(TestingX, Testingy)
        final.append([result,MSE,i])
    return final

def SVMP(letter, f, prep):
    if f == "y":
        TestingX, TrainingX, Trainingy, Testingy = fold(letter)
    elif f == "n":
        TestingX, TrainingX, Trainingy, Testingy = TnT(letter)
    else:
        return "fold can only be either y or n"
    Cs = [0.1,1.0,10.0,100.0,1000.0]
    final = []
    if prep == 's':
        scaler = preprocessing.StandardScaler().fit(TrainingX)
        TrainingX = scaler.transform(TrainingX)
        TestingX = scaler.transform(TestingX)
    elif prep == 'n':
        normalizer = preprocessing.Normalizer().fit(TrainingX)
        TrainingX = normalizer.transform(TrainingX)
        TestingX = normalizer.transform(TestingX)
    for i in Cs:
        fi = make_pipeline(StandardScaler(), SVC(C = i, kernel='poly', gamma='auto'))
        fi.fit(TrainingX, Trainingy)
        MSE = mean_squared_error(Testingy, fi.predict(TestingX))
        result = fi.score(TestingX, Testingy)
        final.append([result,MSE,i])
    return final
# end of machine learning functions #

# ROC, f1, AUC #

def binar(y):
    temp = []
    for i in y:
        if i == 2.0:
            temp.append(1.0)
        else:
            temp.append(0.0)
    return temp

def TnTALT(letter):
    if letter == 'b':
        testing = readFiles(top.groupsFolder+top.breast_testing)
        training = readFiles(top.groupsFolder+top.breast_training)
        data = readFile4(top.groupsFolder+top.breast)
    elif letter == 's':
        testing = readFiles(top.groupsFolder+top.sonar_testing)
        training = readFiles(top.groupsFolder+top.sonar_training)
        data = readFile4(top.groupsFolder+top.sonar)
    elif letter == 'c':
        testing = readFiles(top.groupsFolder+top.covtype_testing)
        training = readFiles(top.groupsFolder+top.covtype_training)
        data = readFileC(top.groupsFolder+top.covtype)
    else:
        return top.error
    if letter == 'b' or letter == 's':
        y, x = getresults(data)
        x = numberfy(x, letter)
    else:
        y, x = covtyperesult(data)
        x = numberfyC(x)
    y = binar(y)
    te = []
    tr = []
    tractual = []
    teactual = []
    for i in testing:
        teactual.append(y[int(i)-1])
        te.append(x[int(i)-1])
    for j in training:
        tractual.append(y[int(j)-1])
        tr.append(x[int(j)-1])
    return te, tr, tractual, teactual

def foldALT(letter):
    if letter == 'b':
        data = readFile4(top.groupsFolder+top.breast)
    elif letter == 's':
        data = readFile4(top.groupsFolder+top.sonar)
    elif letter == 'c':
        data = readFileC(top.groupsFolder+top.covtype)
    else:
        return top.error
    kf = KFold(n_splits=5)
    if letter == 'b' or letter == 's':
        y, X = getresults(data)
        X = numberfy(X, letter)
    else:
        y, X = covtyperesult(data)
        X = numberfyC(X)
    y = binar(y)
    for train_index, test_index in kf.split(X):
        X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
    return X_test, X_train, y_train, y_test

# call this like the previous function but with only 'c' in the letter position #
def computeROC(letter, t, prep):
    if t == "y":
        TestingX, TrainingX, Trainingy, Testingy = foldALT(letter)
    elif t == "n":
        TestingX, TrainingX, Trainingy, Testingy = TnTALT(letter)
    else:
        return "fold can only be either y or n"
    i = 0.1
    if prep == 's':
        scaler = preprocessing.StandardScaler().fit(TrainingX)
        TrainingX = scaler.transform(TrainingX)
        TestingX = scaler.transform(TestingX)
    elif prep == 'n':
        normalizer = preprocessing.Normalizer().fit(TrainingX)
        TrainingX = normalizer.transform(TrainingX)
        TestingX = normalizer.transform(TestingX)
    f = make_pipeline(StandardScaler(), LinearSVC(C = i, dual = False, max_iter = 2000))
    f.fit(TrainingX, Trainingy)
    #y_score = f.decision_function(TestingX)
    fpr, tpr, thresholds = metrics.roc_curve(Testingy, f.predict(TestingX))
    metrics.plot_roc_curve(f, TestingX, Testingy)
    plt.show()
    roc = roc_auc_score(Testingy, f.predict(TestingX))
    result = f.score(TestingX, Testingy)
    score = f1_score(Testingy, f.predict(TestingX))
    return result, score, roc
    