from task_1 import *
import time


def norm_probability(x, mu, std):
    pdf_exp = -1*(((x-mu)**2)/(2*(std**2)))
    prob = (1/(std*np.sqrt(2*np.pi)))*np.exp(pdf_exp)
    return prob

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


# Populate Dictionaries Entries ----------------------------------------------------------------------------------------
a_totals = {}
a_value_count = {}
p_cond = {}
mu_sigma_cont = {}

results_correct = []
results_false = []

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
    else:
        entry_name_cont_mu1 = "a" + str(class_counter) + "_" + "1" + "_" + "mu"
        entry_name_cont_sigma1 = "a" + str(class_counter) + "_" + "1" + "_" + "sigma"
        entry_name_cont_mu2 = "a" + str(class_counter) + "_" + "2" + "_" + "mu"
        entry_name_cont_sigma2 = "a" + str(class_counter) + "_" + "2" + "_" + "sigma"
        mu_sigma_cont[entry_name_cont_mu1] = 0
        mu_sigma_cont[entry_name_cont_mu2] = 0
        mu_sigma_cont[entry_name_cont_sigma1] = 0
        mu_sigma_cont[entry_name_cont_sigma2] = 0

    class_counter += 1

# Populate Dictionaries: attribute counts ------------------------------------------------------------------------------

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


# Populate mu sigma dictionary for continuous values -------------------------------------------------------------------

for i in xrange(train_data.shape[1]):
    if attribute_type[i] == -1:
        key_mu_1 = "a" + str(i) + "_1" + "_mu"
        key_mu_2 = "a" + str(i) + "_2" + "_mu"
        key_sigma_1 = "a" + str(i) + "_1" + "_sigma"
        key_sigma_2 = "a" + str(i) + "_2" + "_sigma"

        column = train_data[:, [i, 15]]
        c1_data = np.array([])
        c2_data = np.array([])

        for row in column:
            if row[1] == 1:
                c1_data = np.append(c1_data, row[0])
            else:
                c2_data = np.append(c2_data, row[0])

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
    print "Accuracy: " + str((1 - float(wrong_predictions) / len(data)) * 100) + "%\n\n"


# Main -----------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # test training set
    print "Bayes classifier on: training set"
    test(train_data)
    print "Bayes classifier on: testing set"
    test(test_data)


