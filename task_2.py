from task_1 import *
import time

# Helper functions -----------------------------------------------------------------------------------------------------


def norm_probability(x, mu, std):
    """
    Normal probability distribution given a standard deviation and mean
    :param x: variable of unknown probability
    :param mu: mean
    :param std: standard deviation
    :return: probability of x
    """
    pdf_exp = -1*(((x-mu)**2)/(2*(std**2)))
    prob = (1/(std*np.sqrt(2*np.pi)))*np.exp(pdf_exp)
    return prob

# Load Data using task 1 -----------------------------------------------------------------------------------------------


df = load_records()
clean_data(df)
attribute_type = get_attributes_np_array(df)
df = map_integers(df)
data_sets = partition_data(df)

# for inspection of data sets
np.savetxt("data/training_set.csv", data_sets[0], delimiter=',', fmt='%10.2f')
np.savetxt("data/testing_set.csv", data_sets[1], delimiter=',', fmt='%10.2f')

train_data = data_sets[0]
test_data = data_sets[1]

# Begin Bayes Classifier construction ----------------------------------------------------------------------------------


num_rows = float(train_data.shape[0])

a15 = (train_data[:, 15]).astype(int)
count_a15 = np.bincount(a15)

# class probabilities
p_a15_1 = count_a15[1]/num_rows
p_a15_2 = count_a15[2]/num_rows

# Populate Dictionaries Keys -------------------------------------------------------------------------------------------


a_totals = {}  # the total counts of each attribute value
a_value_count = {}  # the count of each attribute value associated with class 1 and class 2
p_cond = {}  # contains the conditional probabilities p(attribute_value | class)
mu_sigma_cont = {}  # contains the std-dev and mean values for continuous attributes associated with a particular class

class_counter = 0

# Key notation for each dictionary
# a_value_count:
#   - a4_1_2: count of entries where attribute 4 has a value of 2 and a class value of 2
# a_totals:
#   - a3_1: count of attribute 3 with a value of 1
# p_cond:
#   - a5_10_1: conditional probability of the 5th attribute having a value of 10 and a class value of 1


for type in attribute_type:
    # Check discrete and continuous values
    if type != -1:
        # form dictionary keys
        entry_name_prob_1 = "a" + str(class_counter) + "_" + "1" + "_"  # a_value_count key for class1
        entry_name_prob_2 = "a" + str(class_counter) + "_" + "2" + "_"  # a_value_count key class 2
        entry_name_totals = "a" + str(class_counter) + "_"  # a_totals key
        for i in xrange(1, type + 1):
            totals_entry = entry_name_totals + str(i)
            cond_prob_1 = entry_name_prob_1 + str(i)
            cond_prob_2 = entry_name_prob_2 + str(i)
            # populate with default values of 0
            a_totals[totals_entry] = 0
            a_value_count[cond_prob_1] = 0
            a_value_count[cond_prob_2] = 0
            p_cond[cond_prob_1] = 0
            p_cond[cond_prob_2] = 0
    else:
        # form keys for mu_sigma dictionary for continuous values
        entry_name_cont_mu1 = "a" + str(class_counter) + "_" + "1" + "_" + "mu"
        entry_name_cont_sigma1 = "a" + str(class_counter) + "_" + "1" + "_" + "sigma"
        entry_name_cont_mu2 = "a" + str(class_counter) + "_" + "2" + "_" + "mu"
        entry_name_cont_sigma2 = "a" + str(class_counter) + "_" + "2" + "_" + "sigma"
        mu_sigma_cont[entry_name_cont_mu1] = 0
        mu_sigma_cont[entry_name_cont_mu2] = 0
        mu_sigma_cont[entry_name_cont_sigma1] = 0
        mu_sigma_cont[entry_name_cont_sigma2] = 0

    class_counter += 1

# Populate Dictionary: a_totals ----------------------------------------------------------------------------------------

for i in xrange(train_data.shape[1]):

    # check if data set is discrete
    if attribute_type[i] != -1:
        column = (train_data[:, i]).astype(int)
        # count the unique integer occurrences starting from 0
        column_count = np.bincount(column)

        for j in xrange(1, len(column_count)):
            a_totals_key = "a" + str(i) + "_" + str(j)
            a_totals[a_totals_key] = column_count[j]

# Populate Dictionary: a_value_count -----------------------------------------------------------------------------------

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

# Populate Dictionary: p_cond -----------------------------------------------------------------------------------------

for entry in p_cond:
    # use the p_cond key to access the a_totals and a_value_count dictionaries
    # counts of an attribute value given a class
    a_occurrence = a_value_count[entry]
    entry_string = entry.split("_")
    totals_key = entry_string[0] + "_" + entry_string[2]
    # total occurrence of an attribute value
    a_total = a_totals[totals_key]
    prob_a_given_class = float(a_occurrence)/float(a_total)
    p_cond[entry] = prob_a_given_class

# Populate Dictionary: mu_sigma_cont -----------------------------------------------------------------------------------

for i in xrange(train_data.shape[1]):
    # for all continuous attributes
    if attribute_type[i] == -1:
        # get the sigma and mu for each attribute value for class 1 and class 2
        key_mu_1 = "a" + str(i) + "_1" + "_mu"
        key_mu_2 = "a" + str(i) + "_2" + "_mu"
        key_sigma_1 = "a" + str(i) + "_1" + "_sigma"
        key_sigma_2 = "a" + str(i) + "_2" + "_sigma"

        column = train_data[:, [i, 15]]
        # temp array to hold continuous values of an attribute for a given class
        c1_data = np.array([])
        c2_data = np.array([])

        for row in column:
            if row[1] == 1:
                c1_data = np.append(c1_data, row[0])
            else:
                c2_data = np.append(c2_data, row[0])

        # compute mean and stdev of an attribute for each class
        mu1 = np.mean(c1_data)
        sigma1 = np.std(c1_data)
        mu2 = np.mean(c2_data)
        sigma2 = np.std(c2_data)

        mu_sigma_cont[key_mu_1] = mu1
        mu_sigma_cont[key_mu_2] = mu2
        mu_sigma_cont[key_sigma_1] = sigma1
        mu_sigma_cont[key_sigma_2] = sigma2

# Classify function ----------------------------------------------------------------------------------------------------

def test(data):

    wrong_predictions = 0
    # writes results to a txt file for manual inspection
    results = open("results_" + str(int(time.time())) + ".txt", 'w')
    for row in data:

        # write test case to file
        for attribute in row:
            results.write(str(attribute) + ",")

        p_1 = p_a15_1
        p_2 = p_a15_2
        for i in xrange(len(attribute_type)):
            if i == 15:
                # Check if guessed correctly
                if p_1 > p_2:
                    results.write(" [ + ] ")
                    if row[i] != 1:
                        results.write(" [WRONG]")
                        wrong_predictions += 1
                elif p_2 > p_1:
                    results.write(" [ - ] ")
                    if row[i] != 2:
                        results.write(" [WRONG]")
                        wrong_predictions += 1
                else:
                    results.write(" [ ? ] ")
            if attribute_type[i] == -1:
                # p(a | c) of continuous value
                # get condition probability variables from p_cond
                key_mu_1 = "a" + str(i) + "_1_mu"
                key_sigma_1 = "a" + str(i) + "_1_sigma"
                key_mu_2 = "a" + str(i) + "_2_mu"
                key_sigma_2 = "a" + str(i) + "_2_sigma"

                p_cond_1 = norm_probability(row[i], mu_sigma_cont[key_mu_1], mu_sigma_cont[key_sigma_1])
                p_cond_2 = norm_probability(row[i], mu_sigma_cont[key_mu_2], mu_sigma_cont[key_sigma_2])

                p_1 *= p_cond_1
                p_2 *= p_cond_2

            else:
                # p(a | c) of discrete value
                key_1 = "a" + str(i) + "_1_" + str(int(row[i]))
                key_2 = "a" + str(i) + "_2_" + str(int(row[i]))

                p_cond_1 = p_cond[key_1]
                p_cond_2 = p_cond[key_2]

                p_1 *= p_cond_1
                p_2 *= p_cond_2

        results.write(" [+: " + str(p_1) + "] ")
        results.write(" [-: " + str(p_2) + "] \n")

    print "Wrong results: ", wrong_predictions
    print "Dataset size: ", len(data)
    print "Accuracy: " + str((1 - float(wrong_predictions) / len(data)) * 100) + "%"
    print "Misclassified %: " + str((float(wrong_predictions) / len(data)) * 100) + "%\n\n"


# Main -----------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # test training set
    print "Bayes classifier on: training set"
    test(train_data)
    print "Bayes classifier on: testing set"
    test(test_data)


