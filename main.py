import pandas as pd
import numpy as np

# Pandas DataFrame inspection ------------------------------------------------------------------------------------------

records = pd.read_csv("data/records.txt", header=None)  # pd data frame

# analyse the dtypes
for col in records:
    records[col].head()  # print to manually inspect column types

# col 1 and 13 listed incorrectly as objects are converted to their correct representation (int64 or float64)
incorrect_cols = [1, 13]

for i in incorrect_cols:
    records[i] = pd.to_numeric(records[i], errors='coerce')  # '?' values are replaced with nan

# Mapping nominal values to integers -----------------------------------------------------------------------------------
records_map = {}  # store unique nominal values

for col in records:
    # identify nominal columns as np.objects
    if records[col].dtype == np.object:
        nominal_vals = records[col].unique()

        print type(nominal_vals.toList())
        # print nominal_vals
        records_map[col] = nominal_vals
        # TODO convert nominal_vals to list object

print records_map
# continuous values = np.nt64 / np.float64

