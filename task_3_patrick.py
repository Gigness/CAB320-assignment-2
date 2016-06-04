# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:55:22 2016

@author: Patrick
"""

from task_1 import *

# Constants ------------------------------------------------------------------------------------------------------------
CLASS_INDEX = 15
PLUS = 1
MINUS = 2
MAX_DEPTH = 300000


class Node:
    # init will be used when constructing the tree with training data
    def __init__(self, data, parent=None, depth=0):

        self.parent = parent
        self.depth = depth
        print "New Node at depth ", depth
        print data
        
        self.threshold = 0        
        
        # Get the number of pluses to the number of minuses
        self.plus_minus_ratio = get_plus_minus_ratio(data)

        print "ratio", self.plus_minus_ratio

        # test if it is all pluses and all minuses, and label the Node as such
        self.is_plus_leaf = (self.plus_minus_ratio[1] == 0)
        self.is_minus_leaf = (self.plus_minus_ratio[0] == 0)

        if depth > MAX_DEPTH:
            self.is_plus_leaf = self.plus_minus_ratio[1] <= self.plus_minus_ratio[0]
            self.is_minus_leaf = self.plus_minus_ratio[0] <= self.plus_minus_ratio[1]

        if self.is_plus_leaf:
            print "PLUS LEAF"
        if self.is_minus_leaf:
            print "MINUS LEAF"

        # If it isn't a leaf Node, begin splitting operation
        if not self.is_plus_leaf and not self.is_minus_leaf:

            # We will keep track of which split will yield the highest info gain
            highest_info_gain = 0.0

            # save the highest split as we go
            optimal_split = []

            # save the highest attribute to split on as we go
            self.optimal_attribute = 0

            # loop through each of the 14 attributes
            for index in range(15):

                #Save the threshold to use in the query process
                self.threshold = np.average(data[:,index])
                
                # get an array of 2 numpy arrays. The result of a split by
                # the current attribute
                print "splitting by index: ", index
                split = split_by_attribute(data, index, self.threshold)

                # calculate the information gain of the split
                info_gain = get_info_gain(split, data)

                print "info gain for this split ", info_gain

                # if it is the current highest, save the split
                if info_gain > highest_info_gain:
                    highest_info_gain = info_gain
                    optimal_split = split
                    self.optimal_attribute = index
                
                print "current highest is index ", self.optimal_attribute, " with ig of ", highest_info_gain
        
            print "best vaue to split on: ", self.optimal_attribute
            print "+++++++++++++++"
            print "Creating left Node"
            self.left_Node = Node(optimal_split[0], self, self.depth + 1)
            print "+++++++++++++++"
            print "creating right Node"
            self.right_Node = Node(optimal_split[1], self, self.depth + 1)


    #We will use query when classifying the testing data
    def query(self, data):
        if self.is_plus_leaf:
            return True
        elif self.is_minus_leaf:
            return False
        elif data[self.optimal_attribute] < self.threshold:
            return self.left_Node.query(data)
        elif data[self.optimal_attribute] >= self.threshold:
            return self.left_Node.query(data)
    
    #This will be the function that takes an entire array and outputs
    #an array of 1's or 2's corrosoponding to pluses or minuses
    def test(self, data):
        return 1



def gaussian(values):
    """
    Calculates the probability of each element from an array using a gaussian function
    So far, not needed
    Input: Takes a numpy.array
    output: returns an array of probabilities corrosponding to each of the inputs
    :param values:
    :return:
    """
    mean = np.mean(values)
    stddev = np.std(values)
    var = stddev**2

    pdf_exp = -1 * (((values - mean)**2)/(2 * var))
    probs = (1/(stddev * np.sqrt(2 * np.pi))) * np.exp(pdf_exp)
    return probs


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
    print "ratio in split 1 ", ratio_split1
    ratio_split2 = get_plus_minus_ratio(new_split[1])
    print "ratio in split 2 ", ratio_split2

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

# Testing --------------------------------------------------------------------------------------------------------------

training_data = np.array(([1,30.83,0.0,1,1,1,1,1.25,1,1,1,1,1,202.0,0,1],
    [2,58.67,4.46,1,1,2,2,3.04,1,1,6,1,1,43.0,560,1],
    [2,24.5,0.5,1,1,2,2,1.5,1,2,0,1,1,280.0,824,1],
    [1,27.83,1.54,1,1,1,1,3.75,1,1,5,2,1,100.0,3,1],
    [1,23.42,1.0,1,1,7,1,0.5,2,2,0,2,2,280.0,0,2],
    [2,15.92,2.875,1,1,2,1,0.085,2,2,0,1,1,120.0,0,2],
    [2,24.75,13.665,1,1,2,2,1.5,2,2,0,1,1,280.0,1,2],
    [1,48.75,26.335,2,2,13,4,0.0,1,2,0,2,1,0.0,0,2],
    [2,24.83,4.5,1,1,1,1,1.0,2,2,0,2,1,360.0,6,2],
    [1,19.0,1.75,2,2,7,1,2.335,2,2,0,2,1,112.0,6,2],
    [2,16.33,0.21,1,1,12,1,0.125,2,2,0,1,1,200.0,1,2],
    [2,18.58,10.0,1,1,8,1,0.415,2,2,0,1,1,80.0,42,2],
    [1,18.83,3.54,2,2,13,4,0.0,2,2,0,2,1,180.0,1,2],
    [1,45.33,1.0,1,1,2,1,0.125,2,2,0,2,1,263.0,0,2],
    [2,47.25,0.75,1,1,2,2,2.75,1,1,1,1,1,333.0,892,1],
    [1,24.17,0.875,1,1,2,1,4.625,1,1,2,2,1,520.0,2000,1],
    [1,39.25,9.5,1,1,3,1,6.5,1,1,14,1,1,240.0,4607,1],
    [2,20.5,11.835,1,1,7,2,6.0,1,2,0,1,1,340.0,0,1],
    [2,18.83,4.415,2,2,7,2,3.0,1,2,0,1,1,240.0,0,1],
    [1,19.17,9.5,1,1,1,1,1.5,1,2,0,1,1,120.0,2206,1],
    [2,25.0,0.875,1,1,9,2,1.04,1,2,0,2,1,160.0,5860,1]))

decision_tree = Node(mydata)

testing_data = np.array(([1,41.42,5.0,1,1,2,2,5.0,1,1,6,2,1,470.0,0,1],
                         [2,17.83,11.0,1,1,9,2,1.0,1,1,11,1,1,0.0,3000,1],
                        [1,23.17,11.125,1,1,9,2,0.46,1,1,1,1,1,100.0,0,1],
                        [1,31.5681710914,0.625,1,1,6,1,0.25,2,2,0,1,1,380.0,2010,2],
                        [1,18.17,10.25,1,1,7,2,1.085,2,2,0,1,1,320.0,13,2],
                        [1,20.0,11.045,1,1,7,1,2.0,2,2,0,2,1,136.0,0,2],
                        [1,20.0,0.0,1,1,8,1,0.5,2,2,0,1,1,144.0,0,2],
                        [2,20.75,9.54,1,1,10,1,0.04,2,2,0,1,1,200.0,1000,2],
                        [2,24.5,1.75,2,2,7,1,0.165,2,2,0,1,1,132.0,0,2],
                        [1,32.75,2.335,1,1,8,2,5.75,2,2,0,2,1,292.0,0,2],
                        [2,52.17,0.0,2,2,13,4,0.0,2,2,0,1,1,0.0,0,2]))

print decision_tree.test(test_data)








