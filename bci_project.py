#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 20:32:06 2018

@author: yashadsamant
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn as sk
import vrep
import sys

class BrainComputerInterface():
    def dataPreProcessing(self):
        
        return None
    
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
    BCI = BrainComputerInterface()
    BCI.setConnection() 
    while True:
        BCI.createHandle()     
        
        

    
    

