# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:55:22 2016

@author: Patrick
"""

import pandas as pd
import numpy as np
from task_1 import *

class node:
    def __init__(self, parent=None):
        self.parent = parent
        self.left_node
        self.right_node
        self.threshold
        self.attribute
        
    def split(self):
        self.left_node = node(self)
        self.right_node = node(self)
        return [left_node, right_node]
        
    def query(self, value):
        if value[self.attribute] < self.threshold:
            return True
        else:
            self.child.query(value)
            

def gaussian(values):
    mean = np.mean(values)
    stddev = np.std(values)
    var = stddev**2
    
    print mean
    print stddev
    
    pdf_exp =  -1*(((values-mean)**2)/(2*var)) 
    probs = (1/(stddev*np.sqrt(2*np.pi)))*np.exp(pdf_exp)
    return probs
    

def entropy(values): 
    entropy = np.sum(np.log2(values) * values *-1)
    return np.sum(entropy)


#Find the least entropic attribute, create a node that will query it

#Find the second least entropic attribute, create a node that will query it.

arr = np.array([0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])


gaussian(arr)

pdf_exp =  -1*(((1-2)**2)/(2*0.09)) 
probs = (1/(0.3*np.sqrt(2*np.pi)))*np.exp(pdf_exp)
print probs
