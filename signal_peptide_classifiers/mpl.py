import preprocess_data
import matplotlib.pyplot as plt
from pandas import np
from sklearn.metrics import plot_confusion_matrix
from sklearn.neural_network import MLPClassifier


class MultiLayerPerception:

    def __init__(self):
        self.seq_train = np.array
        self.seq_test = np.array
        self.class_ids_train = list
        self.class_ids_test = list

        self.mlp_classifier = MultiLayerPerception

        self.titles_options = [("Confusion matrix, without normalization", None), ("Normalized confusion matrix", 'true')]

    def pre_process_data(self):

        instances_aa, instances_characteristics, class_ids = preprocess_data.parse_file()
        one_hots = preprocess_data.preprocessing(instances_aa)
        self.seq_train, self.seq_test, self.class_ids_train, self.class_ids_test = preprocess_data.split_data(one_hots, class_ids)

    def classifier(self):

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        self.mlp_classifier = clf.fit(self.seq_train, self.class_ids_train)

    def confusion_matrix(self):

        for title, normalize in self.titles_options:
            disp = plot_confusion_matrix(self.mlp_classifier, self.seq_test, self.class_ids_test,
                                         display_labels=['SP', 'NO SP'],
                                         cmap=plt.cm.Blues,
                                         normalize=normalize)
            disp.figure_.suptitle(title, fontsize=14, verticalalignment='center')
            disp.ax_.set_title(f"Model: {'MLP'}, Data: aminoacid {'identities'}", fontsize=10)

            plt.show()


mlp = MultiLayerPerception()
mlp.pre_process_data()
mlp.classifier()
mlp.confusion_matrix()
