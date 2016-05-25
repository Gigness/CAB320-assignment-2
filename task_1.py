import pandas as pd
import numpy as np

np.set_printoptions(precision=4, suppress=True, threshold=np.inf, linewidth=100)


def load_records():
    """

    :return: pandas DataFrame
    """

    records = pd.read_csv("data/records.txt", header=None)

    # col 1 and 13 listed incorrectly as objects are converted to their correct representation (int64 or float64)
    incorrect_cols = [1, 13]
    for i in incorrect_cols:
        # change data type to numeric values
        records[i] = pd.to_numeric(records[i], errors='coerce')  # '?' values are replaced with NaN
    return records


def clean_data(data_frame):
    """
    
    :param dataframe:
    :return:
    """
    for col in data_frame:

        # if nominal -> replace with most frequent value
        if data_frame[col].dtype == np.object:
            count = data_frame[col].value_counts()
            freq = count.keys()[0]  # grab the most frequent val
            data_frame.ix[data_frame[col] == '?', col] = freq
        # else continuous -> replace with mean

        else:
            mean = data_frame[col].mean()
            data_frame[col] = data_frame[col].fillna(mean)

    # write to file for inspection
    data_frame.to_csv("data/data_frame_cleaned.txt", index=False, header=None)


def get_attributes_np_array(data_frame):
    attribute_set = []
    for col in data_frame:
        if data_frame[col].dtype == np.object:
            attribute_range = len(data_frame[col].unique())
            attribute_set.append(attribute_range)
        else:
            attribute_set.append(-1)
    attribute_set = np.asarray(attribute_set)
    return attribute_set


def map_integers(data_frame):
    """

    :param data_frame:
    :return:
    """
    # store unique nominal values for each column
    data_frame_map = {}
    
    for col in data_frame:
        # identify nominal columns as np.objects
        if data_frame[col].dtype == np.object:
            nominal_vals = data_frame[col].unique()
            mapping = {}
            for i in xrange(len(nominal_vals)):
                mapping[nominal_vals[i]] = i + 1
            data_frame_map[col] = mapping

    for col in data_frame:
        if data_frame[col].dtype == np.object:
            data_frame[col] = data_frame[col].replace(data_frame_map[col])

    # write to file for inspection
    data_frame.to_csv("data/data_frame_cleaned_integered.txt", index=False, header=None)

    # numpy_data = data_frame.as_matrix()
    # np.random.shuffle(numpy_data)
    # return numpy_data
    # data_frame.to_csv("data/data_frame_shuffled.txt", index=False, header=None)
    return data_frame.sample(frac=1)


def partition_data(data_frame):

    partitioned_data = []

    training_set_num = int(len(data_frame.index) * 0.8)
    testing_set_num = len(data_frame.index) - training_set_num

    training_set = data_frame.head(training_set_num)
    testing_set = data_frame.tail(testing_set_num)

    np_training_set = training_set.as_matrix()
    np_testing_set = testing_set.as_matrix()

    partitioned_data.append(np_training_set)
    partitioned_data.append(np_testing_set)

    return partitioned_data


