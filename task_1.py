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
            print nominal_vals
            mapping = {}
            for i in xrange(len(nominal_vals)):
                print i
                mapping[nominal_vals[i]] = i + 1
            data_frame_map[col] = mapping

    for col in data_frame:
        if data_frame[col].dtype == np.object:
            print 'col: ', col, data_frame_map[col]
            data_frame[col] = data_frame[col].replace(data_frame_map[col])

    # write to file for inspection
    data_frame.to_csv("data/data_frame_cleaned_integered.txt", index=False, header=None)

    numpy_data = data_frame.as_matrix()
    np.random.shuffle(numpy_data)
    return numpy_data



