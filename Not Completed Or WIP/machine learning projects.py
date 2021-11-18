# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 18:25:19 2021

@author: rgerr
"""

import numpy as np

class g:
    folder = 'F:/Desktop/competition_hidden_feature/actual/'
    hidden = 'CheXpert_train_hidden_features_all.npy'
    train = 'CheXpert_train_labels_all.npy'
    hiddenlabels = 'CheXpert_valid_hidden_features_all.npy'
    trainlabels = 'CheXpert_valid_labels_all.npy'

def setmainpath(folderpath):
    g.groupsFolder = folderpath

def readFiles(filename):
    data = np.load(filename)
    return data