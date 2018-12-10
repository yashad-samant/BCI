#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 20:32:06 2018

@author: yashadsamant
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing
import vrep
import sys

class BrainComputerInterface():
    def __init__(self, dataName):
        self.data = self.extractData(dataName)
    
    def extractData(self, dataName):
        """
        Input: data.csv
        Output: dataframe
        """
        col = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
               16,17,18,19,20,21,25]
        convert = lambda x: 1 if int(x)==769 else 2 if int(x)==770 else 3 if int(x)==771 else 4 
        data = pd.read_csv(dataName, usecols=col, converters={25:convert})
        data = data.values
        return data
    
    def dataPreProcessing(self):
        """
        input: data(ndarray)
        out: Xtrain,Ytrain,Xtest,Ytest,Xval,Yval
        """
        
        data = self.data
        X = data[:,0:-1]
        
        X = preprocessing.scale(X)
        Y = data[:,-1]
        
        #Randomize dataset
        randi = np.arange(len(X))
        np.random.shuffle(randi)
        X = X[randi]
        print(X.shape)
        Y = Y[randi]
        print(Y.shape)
        
        # Separate data in train,validate and test
        Xtrain = X[0:int(0.6*len(X)),:]; Ytrain = Y[0:int(0.6*len(X))]  
        Xval = X[int(0.6*len(X)):int(0.8*len(X)),:]; Yval = Y[int(0.6*len(X)):int(0.8*len(X))]
        Xtest = X[int(0.8*len(X)):,:]; Ytest = Y[int(0.8*len(X)):]
        
        return X,Y,Xtrain,Ytrain,Xtest,Ytest,Xval,Yval
    
    def setConnection(self):
        vrep.simxFinish(-1) # just in case, close all opened connections

        self.clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5)

        if self.clientID!=-1:  #check if client connection successful
            print ('Connected to remote API server')
    
        else:
            print ('Connection not successful')
            sys.exit('Could not connect')
        
    def createHandle(self):
        joint_h = []
        for x in range(1,6+1):
            errorCode,joint_handle=vrep.simxGetObjectHandle(self.clientID,'Jaco_joint'+str(x),vrep.simx_opmode_oneshot_wait)
            joint_h.append(joint_handle)
        return joint_h

        
if __name__ == "__main__":        
    BCI = BrainComputerInterface('data.csv')
    X,Y,Xtrain,Ytrain,Xtest,Ytest,Xval,Yval = BCI.dataPreProcessing()

    
        
        

    
    

