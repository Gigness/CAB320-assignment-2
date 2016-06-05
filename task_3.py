"""
Created on Tue May 31 16:55:22 2016

@author: Patrick, Paul
"""

from task_1 import *
import numpy as np

    
# Constants ------------------------------------------------------------------------------------------------------------
CLASS_INDEX = 15
# minimum number of rows in a data set to prune the tree
MIN_DATA_PRUNE = 10
# constant for pruning, if ratio of a class in a set exceeds threshold -
# it dominates the set and designates node as a leaf
THRESHOLD_RATIO = 0.75

# Helper Functions -----------------------------------------------------------------------------------------------------


def entropy(ratio):
    """
    Takes an array of probabilities and returns the entropy.

    :param ratio:  A numpy array of probabilities
    :return e: single float representing entropy
    """
    # If any of the values are zero, we know that there is no entropy
    if 0 in ratio:
        return 0

    total = np.sum(ratio)
    probability = ratio/float(total)
    e = np.sum(np.log2(probability) * probability * -1)
    return e


def get_info_gain(new_split, old_data):
    """
    Takes a list of two numpy arrays of split data and a numpy array of the original data
    and calculates the information gain gained from the split
    Only deals with a binary split at the moment, but can be modified

    :param new_split: ([numpy array, numpy array], numpy array)
    :param old_data:
    :return ig: information gain
    """
    ratio_old_data = get_counts(old_data)
    ratio_split1 = get_counts(new_split[0])
    ratio_split2 = get_counts(new_split[1])


    new_entropy = (float(len(new_split[0]))/len(old_data)) * entropy(ratio_split1) +\
                  (float(len(new_split[1]))/len(old_data)) * entropy(ratio_split2)
    ig = entropy(ratio_old_data) - new_entropy
    return ig


def get_counts(data):
    """
    Returns a list of numbers representing plus elements and minus elements in  a given
    set of data.

    Input: a numpy array of data
    ouput: a numpy array of two numbers representing the ratio of pluses to minuses
    :param data: numpy array
    :return np.array: first element is counts of class 1, 2nd element are counts of class 2
    """
    pluses = 0
    minuses = 0
    for each in data:
        if each[CLASS_INDEX] == 1:
            pluses += 1
        else:
            minuses += 1
    return np.array([pluses, minuses])


def split_by_attribute(data, attribute_index, threshold):
    """
    Simply splits a given numpy array of data into two based on the value of the specified index
    Currently, it merely gets the average of the attribute and splits the data in two depending
    on which side it falls on the average. Works for binary values, but may need to be refined
    :param data:
    :param attribute_index:
    :param threshold:
    :return list: A list of two numpy arrays, representing the two sub array that data was split into
    """
    split1 = data[np.where(data[:, attribute_index] < threshold)]
    split2 = data[np.where(data[:, attribute_index] >= threshold)]
    return [split1, split2]

# Decision Node Class --------------------------------------------------------------------------------------------------


class DecisionNode:
    """
    DecisionNode represents a node in the decision tree. Used to build a decison tree in the ID3 Algorithm.
    Instance variables:
    - parent
    - depth
    - ln: left node
    - rn: right node

    """
    
    def __init__(self, depth=0, ln=None, rn=None, optimal_attribute=None, threshold=0, is_plus_leaf=None,
                 is_minus_leaf = None):
        self.depth = depth
        self.ln = None
        self.rn = None
        self.optimal_attribute = None
        self.threshold = threshold
        self.is_plus_leaf = is_plus_leaf
        self.is_minus_leaf = is_minus_leaf

    def query(self, data):
        """
        Tests a single row of data against the decision tree formed with DecisionNodes.
        :param data: single row of data containg all the attributes of the tree
        :return integer: 1 or 2 representing a + or a -
        """
        if self.is_plus_leaf:
            return 1
        elif self.is_minus_leaf:
            return 2
        elif data[self.optimal_attribute] < self.threshold:
            return self.ln.query(data)
        elif data[self.optimal_attribute] >= self.threshold:
            return self.rn.query(data)
    
    def test(self, data):
        """
        Classifies the given data as a 1 (+) or 2 (-) against the decision tree constructed by Decision Nodes.
        :param data: testing data
        :return classify: np.array of 1 or 2 corresponding to the indicies of the input data set.
        """
        classify = np.array([])
        for each in data:
            classify = np.append(classify, self.query(each))
        return classify

# ID3 Implementation ---------------------------------------------------------------------------------------------------


def ID3(data):
    """
    Recrusively builds a tree using Decision Nodes.
    :param data:
    :return:
    """
    root = DecisionNode()
    # counts of each class 1 (+), 2 (-)
    counts = get_counts(data)    

    # if a class dominates entirely, set attribute on node
    root.is_plus_leaf = counts[1] == 0
    root.is_minus_leaf = counts[0] == 0

    # if data set is smaller than threshold, label the data set with the dominating class
    if len(data) < MIN_DATA_PRUNE:
        if counts[0] >= counts[1]:
            root.is_plus_leaf = True
        else:
            root.is_minus_leaf = True
    
    # check if class 1 dominates the data set by given ratio
    if counts[0] > counts[1]:
        ratio_class_1 = float(counts[0]) / (counts[0] + counts[1])
        
        if ratio_class_1 >= THRESHOLD_RATIO:
            root.is_plus_leaf = True

    # check if class 2 dominates the data set by given ratio
    elif counts[1] > counts[0]:
        ratio_class_2 = float(counts[1]) / (counts[0] + counts[1])
        
        if ratio_class_2 >= THRESHOLD_RATIO:
            root.is_minus_leaf = True

    # The data set is not dominates by either class
    # The data set is also not small enough to label the node and terminate
    # Split the data via information gain criterion
    if not root.is_plus_leaf and not root.is_minus_leaf:
            highest_info_gain = 0.0
            # save the highest split as we go
            optimal_split = []
            # loop through each of the 14 attributes
            for index in xrange(CLASS_INDEX):
                # Test for every possible split for the current attribute
                # Get an array of values from the formula x(i) + ( x(i+1) - x(i) )/2
                # The threshold works for continuous and discrete values
                values = np.sort(data[:, index])
                
                operand = np.delete(values, 0)
                operand = np.append(operand, operand[-1])
                threshold_values = ((operand - values) / 2) + values

                # test for every possible split that can be performed
                # for the current attribute column
                for val in threshold_values:

                    split = split_by_attribute(data, index, val)
                    info_gain = get_info_gain(split, data)                  

                    # record attribute which has the current highest info gain
                    # record the threshold value of the attribute and the attribute
                    if info_gain > highest_info_gain:
                        root.threshold = val
                        highest_info_gain = info_gain
                        optimal_split = split
                        root.optimal_attribute = index

            # Create left and right nodes of root node
            left_child = ID3(optimal_split[0])
            left_child.depth = root.depth + 1
            root.ln = left_child
            right_child = ID3(optimal_split[1])
            right_child.depth = root.depth + 1
            root.rn = right_child
            return root
    # terminating root condition, pure collection of classes
    else:

        if root.is_plus_leaf:
            return root
        elif root.is_minus_leaf:
            return root
        else:
            raise ValueError("something bad")

# Testing --------------------------------------------------------------------------------------------------------------


def test_all_data():
    # load and clean data
    df = load_records()
    clean_data(df)
    df = map_integers(df)
    data_sets = partition_data(df)

    train_data = data_sets[0]
    test_data = data_sets[1]

    # build tree
    root = ID3(train_data)

    results = root.test(train_data)
    hit_train = 0
    miss_train = 1

    # Check results of classifier
    for each in range(len(train_data)):
        if train_data[each, 15] == results[each]:
            hit_train += 1
        else:
            miss_train += 1

    accuracy = float(hit_train) / (hit_train + miss_train) * 100
    miss_rate_train = float(miss_train) / (hit_train + miss_train) * 100

    print "Decision Tree Classifier on: training set"
    print "Wrong results: ", miss_train
    print "Dataset size: " + str(len(train_data))
    print "Accuracy: ", accuracy
    print "Misclassified %", miss_rate_train

    results_test = root.test(test_data)
    hit_test, miss_test = [0, 0]

    for each in range(len(test_data)):
        if test_data[each, 15] == results_test[each]:
            hit_test += 1
        else:
            miss_test += 1

    accuracy_test = float(hit_test) / (hit_test + miss_test) * 100
    miss_rate_test = float(miss_test) / (hit_test + miss_test) * 100
    print "\n\nDecision Tree Classifier on: testing set"
    print "Wrong results: ", miss_test
    print "Dataset size: " + str(len(test_data))
    print "Accuracy: ", accuracy_test
    print "Misclassified %", miss_rate_test


if __name__ == '__main__':
    test_all_data()
