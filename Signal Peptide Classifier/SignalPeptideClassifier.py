import re
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid


import matplotlib.pyplot as plt
import logging

logger = logging.getLogger('classifier')
logging.info('Start processing data')
logger.setLevel(logging.DEBUG)

# source: http://www.chem.ucalgary.ca/courses/351/Carey5th/Ch27/ch27-1-4-2.html
# dictionary with pI values per aminoacid
aa_pI = {"G": 5.97, "A": 6.00, "V": 5.96, "L": 5.98, "I": 6.02, "M": 5.74,
         "P": 6.30, "F": 5.48, "W": 5.89, "N": 5.41, "Q": 5.65, "S": 5.68, "T": 5.60,
         "Y": 5.66, "C": 5.07, "D": 2.77, "E": 3.22, "K": 9.74, "R": 10.76, "H": 7.59}


def main():
    train_file, test_file = open_file()
    instances_aa, instances_characteristics, class_ids = parse_file(train_file)

    """
    Coding: identities
    Input for split_data: Aminoacid identities converted to one hot encoded vectors
    This data is splitted into train and test data, and used to fit the following models:
    - Gaussian Naive Bayes
    - SVM
    - Decision tree
    - NearestCentroidClassifier
    """
    coding = 'identities'
    one_hots = preprocessing(instances_aa)
    seq_train, seq_test, class_ids_train, class_ids_test = split_data(one_hots, class_ids)
    gaussian_naive_bayes(seq_train, seq_test, class_ids_train, class_ids_test, coding)
    support_vector_machine(seq_train, seq_test, class_ids_train, class_ids_test, coding)
    decision_tree(seq_train, seq_test, class_ids_train, class_ids_test, coding)
    nearest_centroid_classifier(seq_train, seq_test, class_ids_train, class_ids_test, coding)

    """
    Coding: characteristics
    Input for split_data: Aminoacid characteristics (weight and pI)
    This data is splitted into train and test data, and used to fit the following models:
    - Gaussian Naive Bayes
    - SVM
    - Decision tree
    - NearestCentroidClassifier
    """
    coding = 'characteristics'
    seq_train, seq_test, class_ids_train, class_ids_test = split_data(instances_characteristics, class_ids)
    gaussian_naive_bayes(seq_train, seq_test, class_ids_train, class_ids_test, coding)
    support_vector_machine(seq_train, seq_test, class_ids_train, class_ids_test, coding)
    decision_tree(seq_train, seq_test, class_ids_train, class_ids_test, coding)
    nearest_centroid_classifier(seq_train, seq_test, class_ids_train, class_ids_test, coding)


def open_file():
    train_file = open('train_set.fasta').readlines()
    test_file = open('benchmark_set.fasta').readlines()
    logger.info(' Read training file')
    return train_file, test_file


"""
Input: The content of the opened train file
Function: Reads all the lines of the train file, and searches for aminoacid sequences with the SP and NO-SP label.
Of the sequences that match the pattern, the first 40 aminoacids are saved into a 2D list, 
and the corresponding class-ids (SP/NO-SP) are saved in a list.
In addition, the weight and the pI value of the first 40 aminoacids are saved into a second 2D list.
Output: 2D list of aminoacid identities, 2D list of amicoacid characteristisc, list of the class-ids
"""


def parse_file(train_file):
    instances_aa = []
    instances_characteristics = []
    class_ids = []
    aa = ['A', 'R', 'N', 'D', 'C', 'F', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'P', 'S', 'T', 'W', 'Y',
          'V', 'A', 'R', 'N', 'D', 'C', 'F', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'P', 'S', 'T', 'W', 'Y',
          'V']

    instances_aa.append(aa)

    for index, line in enumerate(train_file):
        match = re.search('\>([A-z0-9]+)\|([A-z]+)\|(NO_SP|SP)', line)
        # matches the following string: (>gene id | organism | NO_SP/SP)

        if match:
            class_id = match.group(3)
            aa_seq = []
            aa_chars = []
            seq = train_file[index + 1].strip()
            counter = 0
            class_ids.append(class_id)
            for amino_acid in seq:
                if counter < 40:
                    aa_seq.append(amino_acid)
                    counter += 1

                    weight = ProteinAnalysis(amino_acid).molecular_weight()  # gets weight in g/mol (molar mass)
                    pI = aa_pI[amino_acid]
                    aa_chars.append(pI)
                    aa_chars.append(weight)

            instances_characteristics.append(aa_chars)
            instances_aa.append(aa_seq)

    return instances_aa, instances_characteristics, class_ids


"""
Input: the 2D list of the aminoacids identities (instances)
Function: converts each instance (in other words: aminoacid sequence), 
to a one hot encoded vector using the OneHotEndoder function from sklearn.
Output: 2D list of one hot encoded vectors
"""


def preprocessing(sequence_list):
    logger.info(' transform data to one hot encoded vectors')
    enc = OneHotEncoder()
    enc.fit(sequence_list)
    one_hots = enc.transform(sequence_list[1:]).toarray()  # type = np.ndarray

    return one_hots


"""
Input: 2D list of one-hot encoded vectors or 2D list of aminoacid characteristics, and the list of class-ids
Function: splits the data into 80% train and 20% test data
Output: seq_train list and seq_test list with the one hot encoded vectors/aminoacids characteristics
class_ids_train list and class_ids_test list with the corresponding class-ids
"""


def split_data(data, class_ids):
    seq_train, seq_test, class_ids_train, class_ids_test = train_test_split(data, class_ids, test_size=0.20,
                                                                            random_state=42)
    return seq_train, seq_test, class_ids_train, class_ids_test


"""
Input: the train data (to fit the model) and test data (for the confusion matrix)
Function: fitting a Gaussion Naive Bayes model with the given data. Calls the confusion_matrix() function.
"""


def gaussian_naive_bayes(seq_train, seq_test, class_ids_train, class_ids_test, coding):
    logger.info(' start fitting gaussian naive bayes model')
    classifier = GaussianNB().fit(seq_train, class_ids_train)
    algorithm = 'Gaussian Naive Bayes'
    confusion_matrix(seq_test, class_ids_test, classifier, algorithm, coding)


"""
Input: the train data (to fit the model) and test data (for the confusion matrix) and the coding (identity versus characteristics)
Function: fitting a Support Vector Machine model with the given data. Calls the confusion_matrix() function.
"""


def support_vector_machine(seq_train, seq_test, class_ids_train, class_ids_test, coding):
    logger.info(' start fitting SVM model')
    classifier = svm.SVC(kernel="rbf").fit(seq_train, class_ids_train)
    algorithm = 'Support Vector Machine'
    confusion_matrix(seq_test, class_ids_test, classifier, algorithm, coding)


"""
Input: the train data (to fit the model) and test data (for the confusion matrix) and the coding (identity versus characteristics)
Function: fitting a Decision tree model with the given data. Calls the confusion_matrix() function.

DecisionTreeClassifier is a class capable of performing multi-class classification on a dataset.

As with other classifiers, DecisionTreeClassifier takes as input two arrays: an array X, sparse or 
dense, of size [n_samples, n_features] holding the training samples, and an array Y of integer values, 
size [n_samples], holding the class labels for the training samples.
"""


def decision_tree(seq_train, seq_test, class_ids_train, class_ids_test, coding):
    logger.info(' start fitting decision tree model')
    classifier = tree.DecisionTreeClassifier().fit(seq_train, class_ids_train)
    algorithm = 'Decision tree'
    confusion_matrix(seq_test, class_ids_test, classifier, algorithm, coding)


"""
Input: the train data (to fit the model) and test data (for the confusion matrix) and the coding (identity versus characteristics)
Function: fitting the nearest_centroid model with the given data. Calls the confusion_matrix() function.

The NearestCentroid classifier is a simple algorithm that represents each class by the centroid of 
its members. In effect, this makes it similar to the label updating phase of the sklearn.cluster.KMeans 
algorithm. It also has no parameters to choose, making it a good baseline classifier. It does, however, 
suffer on non-convex classes, as well as when classes have drastically different variances, as equal 
variance in all dimensions is assumed.
"""


def nearest_centroid_classifier(seq_train, seq_test, class_ids_train, class_ids_test, coding):
    logger.info(' start fitting nearest centroid model')
    classifier = NearestCentroid().fit(seq_train, class_ids_train)
    algorithm = 'Nearest Centroid classifier'
    confusion_matrix(seq_test, class_ids_test, classifier, algorithm, coding)


"""
Input: the test data, the algorithm name and the type of coding (identity versus characteristics)
Function: plot the confusion matrix corresponding with the given test data and model
Output: a matplotlib plot of the calculated confusion matrix
"""


def confusion_matrix(seq_test, class_ids_test, classifier, algorithm, coding):
    # titles_options = [("Confusion matrix, without normalization", None),
    # ("Normalized confusion matrix", 'true')]

    titles_options = [("Normalized confusion matrix", 'true')]

    logger.info(' start making confusion matrix')
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, seq_test, class_ids_test,
                                     display_labels=['SP', 'NO SP'],
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.figure_.suptitle(title, fontsize=14, verticalalignment='center')
        disp.ax_.set_title(f"Model: {algorithm}, Data: aminoacid {coding}", fontsize=10)

        print(title)
        print(disp.confusion_matrix)

    plt.show()


main()
