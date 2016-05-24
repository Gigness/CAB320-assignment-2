# Split the data into training_set and test_set

# ################## Training ####################################

# Get column 0 of all the training_set rows with + and put it into an array

# Get column 0 of all the training_set rows with - and put it into an array

# Calculate the mean and standard deviation of the + data and remember

# Calculate the mean and standard deviation of the - data and remember

# do the same with the rest of the columns

# find the column which has the least overlap between + and - means and standard deviations
# and designate that as the most signficant column, saving its mean and standard deviation

# ########################### Testing ###################################

# get a row from the test_set

# get the value from the most significant column

# put the value through a gaussian function with the + standard deviation and mean to get P(value | +)

# use bayes function to get the P(+ | value) and save it

# put the value through a gaussian function with the - standard deviation and mean to get P(value | -)

# use bayes function to get the P(value | -) and save it

# compare the two probabilities, then assign the row + or - given the most probable

# compare the guessed value of + or - with the real value and save whether its a hit or miss

# do that for the rest of the rows, calculate the accuracy of the algorithm from the hits and misses

import numpy as np
from task_1 import *

df = load_records()

clean_data(df)

df = map_integers(df)

data_sets = partition_data(df)


# print numpy_data

np.savetxt("data/training_set.csv", data_sets[0], delimiter=',', fmt='%10.5f')
np.savetxt("data/testing_set.csv", data_sets[1], delimiter=',', fmt='%10.5f')
