# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:55:22 2016

@author: Patrick
"""

import pandas as pd
import numpy as np
from task_1 import *

#Some constants to improve readability
CLASS_INDEX = 15
PLUS = 1
MINUS = 2

class node:
    #init will be used when constructing the tree with training data
    def __init__(self, data, parent=None):
        
        self.parent = parent
        
        #Get the number of pluses to the number of minuses
        self.plus_minus_ratio = get_plus_minus_ratio(data)
        
        #test if it is all pluses and all minuses, and label the node as such
        self.is_plus_leaf = (self.plus_minus_ratio[1] == 0)
        self.is_minus_leaf = (self.plus_minus_ratio[0] == 0)
        
        #If it isn't a leaf node, begin splitting operation        
        if(not self.is_plus_leaf and not self.is_minus_leaf):

            #We will keep track of which split will yield the highest info gain
            highest_info_gain = 0
            
            #save the highest split as we go            
            optimal_split = []
            
            #loop through each of the attributes 
            for index in range(14):
                
                #get an array of 2 numpy arrays. The result of a split by
                #the current attribute
                split = split_by_attribute(data, index)

                #calculate the information gain of the split
                info_gain = get_info_gain(split, data)

                #if it is the current highest, save the split                
                if(info_gain > highest_info_gain):
                    highest_info_gain == info_gain
                    optimal_split = split
                
            print optimal_split[0]
            print "++++++++++++++++++++++++"
            print optimal_split[1]
            
            
                    


#    def split(self):
#        self.left_node = node(self)
#        self.right_node = node(self)
#        return [left_node, right_node]
#        
#    def query(self, value):
#        if value[self.attribute] < self.threshold:
#            return True
#        else:
#            self.child.query(value)
            

#Calculates the probability of each element from an array using a gaussian function
#So far, not needed
#Input: Takes a numpy.array
#output: returns an array of probabilities corrosponding to each of the inputs
def gaussian(values):
    mean = np.mean(values)
    stddev = np.std(values)
    var = stddev**2
    
    pdf_exp =  -1*(((values-mean)**2)/(2*var)) 
    probs = (1/(stddev*np.sqrt(2*np.pi)))*np.exp(pdf_exp)
    return probs
    

#Takes an array of probabilities and returns the entropy
#Input: A numpy array of probabilities
#Output: a single float representing entropy
def entropy(ratio): 
    total = np.sum(ratio)
    probability = ratio/float(total)
    entropy = np.sum(np.log2(probability) * probability *-1)
    return entropy

#Takes a list of two numpy arrays of split data and a numpy array of the original data
#and calculates the information gain gained from the split
#Only deals with a binary split at the moment, but can be modified
#Input: ([numpy array, numpy array], numpy array)
#Output: a single float representing information gain
def get_info_gain(new_split, old_data):
    ratio_old_data = get_plus_minus_ratio(old_data)
    ratio_split1 = get_plus_minus_ratio(new_split[0])
    ratio_split2 = get_plus_minus_ratio(new_split[1])
    new_entropy = (len(new_split[0])/len(old_data)) * entropy(ratio_split1) + (len(new_split[1])/len(old_data)) * entropy(ratio_split2)
    ig = entropy(ratio_old_data) - new_entropy
    return ig

#Returns a list of numbers representing plus elements and minus elements in  a given
#set of data. Needs to be modified to make it numpy friendly (i.e. uses np.where() etc)
#Input: a numpy array of data
#ouput: a numpy array of two numbers representing the ratio of pluses to minuses
def get_plus_minus_ratio(data):
    pluses = 0
    minuses = 0
    for each in data:
        if each[CLASS_INDEX] == PLUS:
            pluses = pluses+1
        else:
            minuses = minuses+1
    return np.array([pluses, minuses])

#Simply splits a given numpy array of data into two based on the value of the specified index
#Currently, it merely gets the average of the attribute and splits the data in two depending
#on which side it falls on the average. Works for binary values, but may need to be refined
#Input: A numpy array of data and an integer representing to index of the attribute to split on
#Output: A list of two numpy arrays, representing the two sub array that data was split into
def split_by_attribute(data, attribute_index):
    threshold = np.average(data[:,attribute_index])  
    split1 = data[np.where(data[:,attribute_index] < threshold)]
    split2 = data[np.where(data[:,attribute_index] >= threshold)]
    return [split1, split2]
            
            



#######################################Testing#################

mydata = np.array(([1,30.83,0.0,1,1,1,1,1.25,1,1,1,1,1,202.0,0,1],
    [2,58.67,4.46,1,1,2,2,3.04,1,1,6,1,1,43.0,560,1],
    [2,24.5,0.5,1,1,2,2,1.5,1,2,0,1,1,280.0,824,1],
    [1,27.83,1.54,1,1,1,1,3.75,1,1,5,2,1,100.0,3,1],
    [1,23.42,1.0,1,1,7,1,0.5,2,2,0,2,2,280.0,0,2],
    [2,15.92,2.875,1,1,2,1,0.085,2,2,0,1,1,120.0,0,2],
    [2,24.75,13.665,1,1,2,2,1.5,2,2,0,1,1,280.0,1,2],
    [1,48.75,26.335,2,2,13,4,0.0,1,2,0,2,1,0.0,0,2]))

node(mydata)





