import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True, threshold=np.inf, linewidth=100)
# Pandas DataFrame inspection ------------------------------------------------------------------------------------------

records = pd.read_csv("data/records.txt", header=None)  # pd data frame

# for col in records:
#     records[col].head()  # print to manually inspect column types

# col 1 and 13 listed incorrectly as objects are converted to their correct representation (int64 or float64)
incorrect_cols = [1, 13]

for i in incorrect_cols:
    records[i] = pd.to_numeric(records[i], errors='coerce')  # '?' values are replaced with NaN

# Fixing nan values and ? values ---------------------------------------------------------------------------------------

for col in records:

    # if nominal -> replace with most frequent value
    if records[col].dtype == np.object:
        count = records[col].value_counts()
        freq = count.keys()[0]  # grab the most frequent val
        records.ix[records[col] == '?', col] = freq
    # else continuous -> replace with mean

    else:
        mean = records[col].mean()
        records[col] = records[col].fillna(mean)

# write to file for manual inspection
records.to_csv("data/records_cleaned.txt", index=False, header=None)

# Mapping nominal values to integers -----------------------------------------------------------------------------------
records_map = {}  # store unique nominal values

for col in records:
    # identify nominal columns as np.objects
    if records[col].dtype == np.object:
        nominal_vals = records[col].unique()
        print nominal_vals
        mapping = {}
        for i in xrange(len(nominal_vals)):
            print i
            mapping[nominal_vals[i]] = i + 1
        records_map[col] = mapping


for col in records:
    if records[col].dtype == np.object:
        print 'col: ', col, records_map[col]
        records[col] = records[col].replace(records_map[col])

records.to_csv("data/records_cleaned_integered.txt", index=False, header=None)

records_numpy_array = records.as_matrix()
np.random.shuffle(records_numpy_array)

print records_numpy_array

# Code Cemetery --------------------------------------------------------------------------------------------------------

# mean = records[14].mean()
# median = records[14].median()
#
# mean_no_outliers = records[14][records[14] < 10000].mean()
# median_no_outliers = records[14][records[14] < 10000].median()
#
# y_no_outliers = records.ix[records[14] < 10000, 14]
# x_no_outliers = records.ix[records[14] < 10000, 14].keys()
#
# y_outliers = records.ix[records[14] >= 10000, 14]
# x_outliers = records.ix[records[14] >= 10000,14].keys()
#
# plt.plot(x_outliers, y_outliers, 'xr')
# plt.plot(x_no_outliers, y_no_outliers, 'x')
#
# plt.show()