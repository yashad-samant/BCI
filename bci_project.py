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
from sklearn.decomposition import PCA
import vrep
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    
    def findIndex(self,l,e):
        l = l.tolist()
        return [index for index, value in enumerate(l) if value == e] 
   
    def replaceNAN(self, X, Y):
        """
        input: X
        output: replaced mean value for NaN
        """
        
        b = np.isnan(X)
        for i in range(len(b)):
            if True in b[i]:
                rows = self.findIndex(Y, Y[i])
                cols = self.findIndex(b[i],True) 
                rows.remove(i)                 
                for c in cols:
                    X[i,c] = X[rows,c].mean()
        return X
    
    def dataPreProcessing(self):
        """
        input: data(ndarray)
        out: Xtrain,Ytrain,Xtest,Ytest,Xval,Yval
        """
        
        data = self.data
        X = data[:,0:-1]
        Y = data[:,-1]
        X = self.replaceNAN(X,Y)
        X = preprocessing.scale(X)
                
        #Randomize dataset
        randi = np.arange(len(X))
        np.random.shuffle(randi)
        X = X[randi]
        Y = Y[randi]
        
        # Separate data in train,validate and test
        Xtrain = X[0:int(0.6*len(X)),:]; Ytrain = Y[0:int(0.6*len(X))]  
        Xval = X[int(0.6*len(X)):int(0.8*len(X)),:]; Yval = Y[int(0.6*len(X)):int(0.8*len(X))]
        Xtest = X[int(0.8*len(X)):,:]; Ytest = Y[int(0.8*len(X)):]
        
        return X,Y,Xtrain,Ytrain,Xtest,Ytest,Xval,Yval
    
    def PCA(self, X, Y, N):
        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(X)
#        df = pd.DataFrame({'var':pca.explained_variance_ratio_,
#             'PC':['PC1','PC2','PC3']})
#        sns.barplot(x='PC',y="var", 
#           data=df, color="c");
        
        principalDF = np.column_stack((principalComponents,Y))
        #print(finalDF.shape)
        
        finalDf = pd.DataFrame(data = principalDF
             , columns = ['PC1', 'PC2', 'PC3', 'target'])
        
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(111, projection='3d') 
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_ylabel('Principal Component 3', fontsize = 15)
        ax.set_title('3 component PCA', fontsize = 20)
        
        targets = [1,2,3,4]
        colors = ['r', 'g', 'b','y']
        #return finalDf
        for target, color in zip(targets,colors):
                indicesToKeep = finalDf['target'] == target
                ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
               , finalDf.loc[indicesToKeep, 'PC2'], finalDf.loc[indicesToKeep, 'PC3']
               , c = color
               , s = 150)
        ax.legend(targets)
        ax.grid()
#        
        
    
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
    DF = BCI.PCA(X,Y,2)


    
        
        

    
    

