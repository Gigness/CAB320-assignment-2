# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:55:22 2016

@author: Patrick
"""

from task_1 import *
import numpy as np

    

# Constants ------------------------------------------------------------------------------------------------------------
CLASS_INDEX = 15
PLUS = 1
MINUS = 2
MAX_DEPTH = 300000


    #If the tree reached its maximum depth, designate it as a plas or minus
    #depending on which the mode of the data
#    if depth > MAX_DEPTH:
#        self.is_plus_leaf = self.plus_minus_ratio[1] <= self.plus_minus_ratio[0]
#        self.is_minus_leaf = self.plus_minus_ratio[0] <= self.plus_minus_ratio[1]
#    # If it isn't a leaf Node, begin splitting operation
#    if not self.is_plus_leaf and not self.is_minus_leaf:



class Node:
    # init will be used when constructing the tree with training data
    def __init__(self, data, parent=None, depth=0):
        self.parent = parent
        self.depth = depth
        self.threshold = 0     
        self.optimal_attribute = 0
        # Get the number of pluses to the number of minuses
        self.plus_minus_ratio = get_plus_minus_ratio(data)
        # test if it is all pluses and all minuses, and label the Node as such
        self.is_plus_leaf = (self.plus_minus_ratio[1] == 0)
        self.is_minus_leaf = (self.plus_minus_ratio[0] == 0)
        
        #If the tree reached its maximum depth, designate it as a plas or minus
        #depending on which the mode of the data
        if depth > MAX_DEPTH:
            self.is_plus_leaf = self.plus_minus_ratio[1] <= self.plus_minus_ratio[0]
            self.is_minus_leaf = self.plus_minus_ratio[0] <= self.plus_minus_ratio[1]
        # If it isn't a leaf Node, begin splitting operation
        if not self.is_plus_leaf and not self.is_minus_leaf:
            # We will keep track of which split will yield the highest info gain
            highest_info_gain = 0.0
            # save the highest split as we go
            optimal_split = []
            # loop through each of the 14 attributes
            for index in range(15):

                #Test for every possible split on continuous values
                #Get an array of values x(i) + ( x(i+1) - x(i) )/2
                values = np.sort(data[:,index])
                operand = np.delete(values, 0)
                operand = np.append(operand, operand[-1])
                threshold_values = ((operand - values)/2)+ values
                
                for thresh in threshold_values:

                    split = split_by_attribute(data, index, thresh)
                    info_gain = get_info_gain(split, data)                  
                    
                    if info_gain > highest_info_gain:
                        self.threshold = thresh
                        highest_info_gain = info_gain
                        optimal_split = split
                        self.optimal_attribute = index

#                #get threshold from which we split
#                index_threshold = np.average(data[:,index])
#                
#                # get an array of 2 numpy arrays. The result of a split by
#                # the current attribute
#                split = split_by_attribute(data, index, index_threshold)
#                # calculate the information gain of the split
#                info_gain = get_info_gain(split, data)
#                # if it is the current highest, save the split
#                if info_gain > highest_info_gain:
#                    self.threshold = index_threshold
#                    highest_info_gain = info_gain
#                    optimal_split = split
#                    self.optimal_attribute = index

            self.left_Node = Node(optimal_split[0], self, self.depth + 1)
            self.right_Node = Node(optimal_split[1], self, self.depth + 1)


    #We will use query when classifying the testing data
    def query(self, data):
        if self.is_plus_leaf:
            return 1
        elif self.is_minus_leaf:
            return 2
        elif data[self.optimal_attribute] < self.threshold:
            return self.left_Node.query(data)
        elif data[self.optimal_attribute] >= self.threshold:
            return self.right_Node.query(data)
    
    #This will be the function that takes an entire array and outputs
    #an array of 1's or 2's corrooponding to pluses or minuses
    def test(self, data):
        classify = np.array([])
        for each in data:
            classify = np.append(classify, self.query(each))
        return classify





def entropy(ratio):
    """
    Takes an array of probabilities and returns the entropy
    Input: A numpy array of probabilities
    Output: a single float representing entropy
    :param ratio:
    :return:
    """
    # If any of the values are zero, we know that there is no entropy
    if 0 in ratio:
        return 0

    total = np.sum(ratio)
    probability = ratio/float(total)
    entropy = np.sum(np.log2(probability) * probability *-1)
    return entropy


def get_info_gain(new_split, old_data):
    """
    Takes a list of two numpy arrays of split data and a numpy array of the original data
    and calculates the information gain gained from the split
    Only deals with a binary split at the moment, but can be modified
    Input: ([numpy array, numpy array], numpy array)
    Output: a single float representing information gain
    :param new_split:
    :param old_data:
    :return:
    """
    ratio_old_data = get_plus_minus_ratio(old_data)
    ratio_split1 = get_plus_minus_ratio(new_split[0])
    ratio_split2 = get_plus_minus_ratio(new_split[1])


    new_entropy = (float(len(new_split[0]))/len(old_data)) * entropy(ratio_split1) +\
                  (float(len(new_split[1]))/len(old_data)) * entropy(ratio_split2)
    ig = entropy(ratio_old_data) - new_entropy
    return ig


def get_plus_minus_ratio(data):
    """
    Returns a list of numbers representing plus elements and minus elements in  a given
    set of data. Needs to be modified to make it numpy friendly (i.e. uses np.where() etc)
    Input: a numpy array of data
    ouput: a numpy array of two numbers representing the ratio of pluses to minuses
    :param data:
    :return:
    """
    pluses = 0
    minuses = 0
    for each in data:
        if each[CLASS_INDEX] == PLUS:
            pluses = pluses+1
        else:
            minuses = minuses+1
    return np.array([pluses, minuses])


def split_by_attribute(data, attribute_index, threshold):
    """
    Simply splits a given numpy array of data into two based on the value of the specified index
    Currently, it merely gets the average of the attribute and splits the data in two depending
    on which side it falls on the average. Works for binary values, but may need to be refined
    Input: A numpy array of data and an integer representing to index of the attribute to split on
    Output: A list of two numpy arrays, representing the two sub array that data was split into
    :param data:
    :param attribute_index:
    :return:
    """
    split1 = data[np.where(data[:,attribute_index] < threshold)]
    split2 = data[np.where(data[:,attribute_index] >= threshold)]
    return [split1, split2]



                        
df = load_records()

clean_data(df)

attribute_type = get_attributes_np_array(df)

df = map_integers(df)

data_sets = partition_data(df)

np.savetxt("data/training_set.csv", data_sets[0], delimiter=',', fmt='%10.2f')
np.savetxt("data/testing_set.csv", data_sets[1], delimiter=',', fmt='%10.2f')

train_data = data_sets[0]
test_data = data_sets[1]

class DecisionNode:
    
    def __init__(self, parent=None, depth=0, ln=None, rn=None):
        self.parent = parent
        self.depth = depth
        self.ln = None
        self.rn = None
        
        #We will use query when classifying the testing data
    def query(self, data):
        if self.is_plus_leaf:
            return 1
        elif self.is_minus_leaf:
            return 2
        elif data[self.optimal_attribute] < self.threshold:
            return self.ln.query(data)
        elif data[self.optimal_attribute] >= self.threshold:
            return self.rn.query(data)
    
    #This will be the function that takes an entire array and outputs
    #an array of 1's or 2's corrooponding to pluses or minuses
    def test(self, data):
        classify = np.array([])
        for each in data:
            classify = np.append(classify, self.query(each))
        return classify

def ID3(data):
    root = DecisionNode()
    plus_minus_ratio = get_plus_minus_ratio(data)    
    
    root.is_plus_leaf = plus_minus_ratio[1] == 0
    root.is_minus_leaf = plus_minus_ratio[0] == 0
    
    print root.is_plus_leaf
    print root.is_minus_leaf
    
    if not root.is_plus_leaf and not root.is_minus_leaf:
            highest_info_gain = 0.0
            # save the highest split as we go
            optimal_split = []
            # loop through each of the 14 attributes
            for index in xrange(CLASS_INDEX):
                #Test for every possible split on continuous values
                #Get an array of values x(i) + ( x(i+1) - x(i) )/2
                values = np.sort(data[:,index])
                operand = np.delete(values, 0)
                operand = np.append(operand, operand[-1])
                threshold_values = ((operand - values)/2)+ values
                
                for thresh in threshold_values:

                    split = split_by_attribute(data, index, thresh)
                    info_gain = get_info_gain(split, data)                  
                    
                    if info_gain > highest_info_gain:
                        root.threshold = thresh
                        highest_info_gain = info_gain
                        optimal_split = split
                        root.optimal_attribute = index
            
            left_child = ID3(optimal_split[0])
            left_child.depth = root.depth + 1
            root.ln = left_child
            right_child = ID3(optimal_split[1])
            right_child.depth = root.depth + 1
            root.rn = right_child
            return root
    else:
        if root.is_plus_leaf:
            return root
        elif root.is_minus_leaf:
            return root
        else:
            raise ValueError("something bad")

root = ID3(test_data)

results = root.test(train_data)
hit, miss = [0, 0]

for each in range(len(train_data)):
    if train_data[each, 15] == results[each]:
        hit += 1
    else:
        miss += 1

print hit, " hits and ", miss, " misses with a total accuracuy of %", (float(hit)/len(train_data))*100


#
#
#decision_tree = Node(train_data)
#results = decision_tree.test(train_data)
#
#
#hit = 0
#miss = 0
#
#for each in range(len(train_data)):
#    if train_data[each, 15] == results[each]:
#        hit += 1
#    else:
#        miss += 1
#
#print hit, " hits and ", miss, " misses with a total accuracuy of %", (float(hit)/len(train_data))*100










