import re
import matplotlib.pyplot as plt
import logging
import preprocess_data
from sklearn.metrics import plot_confusion_matrix
from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


logger = logging.getLogger('classifier')
logging.info('Start processing data')
logger.setLevel(logging.DEBUG)

# source: http://www.chem.ucalgary.ca/courses/351/Carey5th/Ch27/ch27-1-4-2.html
# dictionary with pI values per amino acid
aa_pI = {"G": 5.97, "A": 6.00, "V": 5.96, "L": 5.98, "I": 6.02, "M": 5.74,
         "P": 6.30, "F": 5.48, "W": 5.89, "N": 5.41, "Q": 5.65, "S": 5.68, "T": 5.60,
         "Y": 5.66, "C": 5.07, "D": 2.77, "E": 3.22, "K": 9.74, "R": 10.76, "H": 7.59}


def main():
    instances_aa, instances_characteristics, class_ids = preprocess_data.parse_file()
    one_hots = preprocess_data.preprocessing(instances_aa)

    """
    Coding: identities
    Input for split_data: Aminoacid identities converted to one hot encoded vectors
    This data is splitted into train and test data, and used to fit the following models:
    - Gaussian Naive Bayes
    - Stochastic Gradient Descent
    - Support Vector Machine
    - Decision Tree
    - Nearest Centroid Classifier
    - Neural Network Multi-layer Perception
    - Ensemble Random Forest Classifier
    """
    coding = 'identities'
    seq_train, seq_test, class_ids_train, class_ids_test = preprocess_data.split_data(one_hots, class_ids)
    gaussian_naive_bayes(seq_train, seq_test, class_ids_train, class_ids_test, coding)
    support_vector_machine(seq_train, seq_test, class_ids_train, class_ids_test, coding)
    decision_tree(seq_train, seq_test, class_ids_train, class_ids_test, coding)
    nearest_centroid_classifier(seq_train, seq_test, class_ids_train, class_ids_test, coding)
    multi_layer_perceptron(seq_train, seq_test, class_ids_train, class_ids_test, coding)
    random_forest_classifier(seq_train, seq_test, class_ids_train, class_ids_test, coding)
    stochastic_gradient_descent(seq_train, seq_test, class_ids_train, class_ids_test, coding)

    """
    Coding: characteristics
    Input for split_data: Aminoacid characteristics (weight and pI)
    This data is splitted into train and test data, and used to fit the following models:
    - Gaussian Naive Bayes
    - Stochastic Gradient Descent
    - Support Vector Machine
    - Decision Tree
    - Nearest Centroid Classifier
    - Neural Network Multi-layer Perception
    - Ensemble Random Forest Classifier
    """
    coding = 'characteristics'
    seq_train, seq_test, class_ids_train, class_ids_test = preprocess_data.split_data(instances_characteristics, class_ids)
    gaussian_naive_bayes(seq_train, seq_test, class_ids_train, class_ids_test, coding)
    support_vector_machine(seq_train, seq_test, class_ids_train, class_ids_test, coding)
    decision_tree(seq_train, seq_test, class_ids_train, class_ids_test, coding)
    nearest_centroid_classifier(seq_train, seq_test, class_ids_train, class_ids_test, coding)
    multi_layer_perceptron(seq_train, seq_test, class_ids_train, class_ids_test, coding)
    random_forest_classifier(seq_train, seq_test, class_ids_train, class_ids_test, coding)
    stochastic_gradient_descent(seq_train, seq_test, class_ids_train, class_ids_test, coding)


"""
Input: the train data (to fit the model) and test data (for the confusion matrix) and the coding (identity versus characteristics)
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
Function: fitting a Stochastic Gradient Descent model with the given data. Calls the confusion_matrix() function.
"""
def stochastic_gradient_descent(seq_train, seq_test, class_ids_train, class_ids_test, coding):

    logger.info(' start fitting SGD model')
    if coding == 'identities':
        classifier = SGDClassifier(loss='log', penalty='l2', max_iter=12)
        # the following parameters are set to fit the optimal model and produce the optimal confusion matrix
        # loss='log' (loss function, tried: hinge, modified_huber and log)
        # penalty='l2' (L2 regularization)
        # max_iter=12 (maximum of iterations in combination with the log loss function, tried: 5, 8, 10, 12, 15 and 20)

    elif coding == 'characteristics':
        classifier = SGDClassifier(loss='log', penalty='l2', max_iter=10)
        # the following parameters are set to fit the optimal model and produce the optimal confusion matrix
        # loss='log' (loss function, tried: hinge, modified_huber and log)
        # penalty='l2' (L2 regularization)
        # max_iter=10 (maximum of iterations in combination with the log loss function, tried: 5, 8, 10, 12, 15, 18 and 20)

    classifier.fit(seq_train, class_ids_train)
    algorithm = 'Stochastic Gradient Descent'
    confusion_matrix(seq_test, class_ids_test, classifier, algorithm, coding)


"""
Input: the train data (to fit the model) and test data (for the confusion matrix) and the coding (identity versus characteristics)
Function: fitting a (Neural Network) Multi Layer Perception model with the given data. Calls the confusion_matrix() function.
"""
def multi_layer_perceptron(seq_train, seq_test, class_ids_train, class_ids_test, coding):

    logger.info(f' start fitting MLP model')
    classifier = MLPClassifier().fit(seq_train, class_ids_train)
    algorithm = 'Neural Network Multi Layer Perceptron'
    confusion_matrix(seq_test, class_ids_test, classifier, algorithm, coding)


"""
Input: the train data (to fit the model) and test data (for the confusion matrix) and the coding (identity versus characteristics)
Function: fitting a (Ensemble) Random Forest Classifier Model with the given data. Calls the confusion_matrix() function.
"""
def random_forest_classifier(seq_train, seq_test, class_ids_train, class_ids_test, coding):

    logger.info(f' start fitting RFC model')
    classifier = RandomForestClassifier().fit(seq_train, class_ids_train)
    algorithm = 'Ensemble Random Forest Classifier'
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
    classifier = tree.DecisionTreeClassifier(max_depth=5, criterion='entropy', class_weight='balanced', min_impurity_decrease=0.01).fit(seq_train, class_ids_train)
    algorithm = 'Decision tree'
    confusion_matrix(seq_test, class_ids_test, classifier, algorithm, coding)

    if coding == 'characteristics':
        feature_names_char = 40*['pI', 'weight']
        plt.figure(figsize=(20, 12))
        tree.plot_tree(classifier,
              feature_names=feature_names_char,
              class_names=['SP', 'NO SP'],
              filled=True,
              rounded=True,
              fontsize=12)
        plt.show()



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

    titles_options = [("Confusion matrix, without normalization", None), ("Normalized confusion matrix", 'true')]

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
