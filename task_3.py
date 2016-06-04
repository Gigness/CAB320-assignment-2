# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:55:22 2016

@author: Patrick
"""

from task_1 import *

# Constants ------------------------------------------------------------------------------------------------------------
CLASS_INDEX = 15
CLASS_INDEX_TEST = 4
PLUS = 1
MINUS = 2
MAX_DEPTH = 3


def ID3(data, attributes):
    [c1_count, c2_count] = get_plus_minus_ratio(data)

    root = TreeNode()

    if c1_count == len(data):
        # Check for Pure partition
        root.classification = 1
        return root
    elif c2_count == len(data):
        root.classification = 2
        return root
    elif len(attributes) == 0:
        # check if no more attributes to split on
        # return leaf node with the highest count of classifications
        class_count = data[:, CLASS_INDEX_TEST]
        _, c1_count, c2_count = np.bincount(class_count)
        if c1_count >= c2_count:
            root.classification = 1
        elif c2_count > c1_count:
            root.classification = 2
        if root is None:
            raise ValueError("root is null")
        return root
    else:
        # Get attribute with highest information gain

        pass


class TreeNode:

    def __init__(self, attribute=None, value=None, children=None, depth=0, classification=None):
        self.attribute = attribute
        self.value = value
        self.children = children
        self.depth = depth
        self.classification = classification

    def __repr__(self):
        return "<Node attribute: %s, value: %s, classification: %s>" % (self.attribute, self.value, self.classification)

        

class Node:
    # init will be used when constructing the tree with training data
    def __init__(self, data, parent=None, depth=0):

        self.parent = parent
        self.depth = depth
        print "New Node at depth ", depth
        print data
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
            print "PLUS LEAF"

        # If it isn't a leaf Node, begin splitting operation
        if not self.is_plus_leaf and not self.is_minus_leaf:

            # We will keep track of which split will yield the highest info gain
            highest_info_gain = 0

            # save the highest split as we go
            optimal_split = []

            # save the highest split as we go
            self.optimal_attribute = 0

            # loop through each of the 14 attributes
            for index in range(15):

                # get an array of 2 numpy arrays. The result of a split by
                # the current attribute
                print "splitting by index: ", index
                split = split_by_attribute(data, index)

                # calculate the information gain of the split
                info_gain = get_info_gain(split, data)

                print "info gain for this split ", info_gain

                # if it is the current highest, save the split
                if info_gain > highest_info_gain:
                    highest_info_gain == info_gain
                    optimal_split = split
                    self.optimal_attribute = index
            print "best vaue to split on: ", self.optimal_attribute
            print "+++++++++++++++"
            print "Creating left Node"
            self.left_Node = Node(optimal_split[0], self, self.depth + 1)
            print "+++++++++++++++"
            print "creating right Node"
            self.right_Node = Node(optimal_split[1], self, self.depth + 1)

#     def split(self):
#         self.left_Node = Node(self)
#         self.right_Node = Node(self)
#         return [left_Node, right_Node]
#
#     def query(self, value):
#         if value[self.attribute] < self.threshold:
#             return True
#         else:
#             self.child.query(value)


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
        if each[CLASS_INDEX_TEST] == PLUS:
            pluses += 1
        else:
            minuses += 1
    return np.array([pluses, minuses])


def split_by_attribute(data, attribute_index):
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
    threshold = np.average(data[:,attribute_index])
    split1 = data[np.where(data[:,attribute_index] < threshold)]
    split2 = data[np.where(data[:,attribute_index] >= threshold)]
    return [split1, split2]

# Testing --------------------------------------------------------------------------------------------------------------

mydata = np.array((
    [1, 30.83, 0.0, 1, 1, 1, 1, 1.25, 1, 1, 1, 1, 1, 202.0, 0, 1],
    [2, 58.67, 4.46, 1, 1, 2, 2, 3.04, 1, 1, 6, 1, 1, 43.0, 560, 1],
    [2, 24.5, 0.5, 1, 1, 2, 2, 1.5, 1, 2, 0, 1, 1, 280.0, 824, 1],
    [1, 27.83, 1.54, 1, 1, 1, 1, 3.75, 1, 1, 5, 2, 1, 100.0, 3, 1],
    [1, 23.42, 1.0, 1, 1, 7, 1, 0.5, 2, 2, 0, 2, 2, 280.0, 0, 2],
    [2, 15.92, 2.875, 1, 1, 2, 1, 0.085, 2, 2, 0, 1, 1, 120.0, 0, 2],
    [2, 24.75, 13.665, 1, 1, 2, 2, 1.5, 2, 2, 0, 1, 1, 280.0, 1, 2],
    [1, 48.75, 26.335, 2, 2, 13, 4, 0.0, 1, 2, 0, 2, 1, 0.0, 0, 2]))

# Testing data
sunny = 1
overcast = 2
rain = 3
hot = 1
mild = 2
cool = 3
high = 1
normal = 2
weak = 1
strong = 2
yes = 1
no = 2

example_data = np.array([
    [sunny, hot, high, weak, no],
    [sunny, hot, high, strong , no],
    [overcast, hot, high, weak, yes],
    [rain, mild, high, weak, yes],
    [rain, cool, normal, weak, yes],
    [rain, cool, normal, strong, no],
    [overcast, cool, normal, strong, yes],
    [sunny, mild, high, weak, no],
    [sunny, cool, normal, weak, yes],
    [rain, mild, normal, weak, yes],
    [sunny, mild, normal, strong, yes],
    [overcast, mild, high, strong, yes],
    [overcast, hot, normal, weak, yes],
    [rain, mild, high, strong, no],
])

# decision_tree = Node(mydata)

node = ID3(example_data, [])

print node

