from task_1 import *
import scipy  # for gaussian model of continuous variables

df = load_records()

clean_data(df)

attribute_type = get_attributes_np_array(df)


df = map_integers(df)

data_sets = partition_data(df)

np.savetxt("data/training_set.csv", data_sets[0], delimiter=',', fmt='%10.2f')
np.savetxt("data/testing_set.csv", data_sets[1], delimiter=',', fmt='%10.2f')

train_data = data_sets[0]
test_data = data_sets[1]

if False:
    # Testing numpy array access
    # access an entire column via splciing

    matrix = np.diag([1, 2, 3, 4])
    print matrix
    print matrix[:, 3]
    print matrix[:, 1]

    a = np.array([0, 0, 1, 1])

    a_class_1 = 0
    a_class_2 = 0
    a_class = (train_data[:, 15])

    a_1 = (train_data[:, [0, 15]])

    print a_1

    a_class = a_class.astype(int)
    print a_class

    print np.bincount(a_class)

# building the classifier
num_rows = float(train_data.shape[0])

a15 = (train_data[:, 15]).astype(int)

count_a15 = np.bincount(a15)

# class probabilities
p_a15_1 = count_a15[1]/num_rows
p_a15_2 = count_a15[2]/num_rows

# probabilities for a0
a0 = (train_data[:, [0, 15]]).astype(int)

# Populate Dictionaries Entries ----------------------------------------------------------------------------------------
a_totals = {}
a_value_count = {}
p_cond = {}

class_counter = 0
for type in attribute_type:
    if type != -1:
        entry_name_prob_1 = "a" + str(class_counter) + "_" + "1" + "_"
        entry_name_prob_2 = "a" + str(class_counter) + "_" + "2" + "_"
        entry_name_totals = "a" + str(class_counter) + "_"
        for i in xrange(1, type + 1):
            totals_entry = entry_name_totals + str(i)
            cond_prob_1 = entry_name_prob_1 + str(i)
            cond_prob_2 = entry_name_prob_2 + str(i)
            a_totals[totals_entry] = 0
            a_value_count[cond_prob_1] = 0
            a_value_count[cond_prob_2] = 0
            p_cond[cond_prob_1] = 0
            p_cond[cond_prob_2] = 0
            # print totals_entry
    class_counter += 1

# Populate Dictionaries Values -----------------------------------------------------------------------------------------

for i in xrange(train_data.shape[1]):

    if attribute_type[i] != -1:
        column = (train_data[:, i]).astype(int)
        column_count = np.bincount(column)

        for j in xrange(1, len(column_count)):
            a_totals_key = "a" + str(i) + "_" + str(j)
            a_totals[a_totals_key] = column_count[j]


# Populate counts of each attribute ------------------------------------------------------------------------------------

for i in xrange(train_data.shape[1]):

    if attribute_type[i] != -1:

        attr_and_class = (train_data[:, [i, 15]]).astype(int)
        a_value_count_key1 = "a" + str(i) + "_" + "1" + "_"
        a_value_count_key2 = "a" + str(i) + "_" + "2" + "_"

        for row in attr_and_class:
            if row[1] == 1:
                a_value_count_key = a_value_count_key1 + str(row[0])
                a_value_count[a_value_count_key] += 1
            else:
                a_value_count_key = a_value_count_key2 + str(row[0])
                a_value_count[a_value_count_key] += 1

# Populate conditional probability -------------------------------------------------------------------------------------

for entry in p_cond:
    a_occurrence = a_value_count[entry]
    entry_string = entry.split("_")
    totals_key = entry_string[0] + "_" + entry_string[2]
    a_total = a_totals[totals_key]
    prob_a_given_class = float(a_occurrence)/float(a_total)
    p_cond[entry] = prob_a_given_class




print attribute_type


