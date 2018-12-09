#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 11:25:43 2018

@author: yashadsamant
"""

list1 = [[0,1,2,3], [0,1,2], [0,1], [0], []]
list2 = [[], [3], [2,3], [1,2,3], [0,1,2,3]]
A = []

i = 0
for i in range(len(list1)-1):
    A.append(list1[i]+list2[i])