import preprocess_data
import matplotlib.pyplot as plt
from pandas import np
from sklearn.metrics import plot_confusion_matrix
from sklearn.neural_network import MLPClassifier


class MultiLayerPerception:

    def __init__(self):
        self.instances_aa = list
        self.instances_characteristics = list
        self.class_ids = list
        self.one_hots = np.array

        self.seq_train = np.array
        self.seq_test = np.array
        self.class_ids_train = list
        self.class_ids_test = list

        self.mlp_classifier = MultiLayerPerception

        self.titles_options = [("Confusion matrix, without normalization", None),
                               ("Normalized confusion matrix", 'true')]

    def parse_file(self):
        self.instances_aa, self.instances_characteristics, self.class_ids = preprocess_data.parse_file()

    def pre_process_identities_data(self):
        self.one_hots = preprocess_data.preprocessing(self.instances_aa)

    def split_data(self, coding):

        if coding == 'identities':
            self.seq_train, self.seq_test, self.class_ids_train, self.class_ids_test = preprocess_data.split_data(self.one_hots, self.class_ids)
        else:
            self.seq_train, self.seq_test, self.class_ids_train, self.class_ids_test = preprocess_data.split_data(self.instances_characteristics, self.class_ids)

    def classifier(self):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2,), random_state=2)
        self.mlp_classifier = clf.fit(self.seq_train, self.class_ids_train)

    def confusion_matrix(self, coding):
        for title, normalize in self.titles_options:
            disp = plot_confusion_matrix(self.mlp_classifier, self.seq_test, self.class_ids_test,
                                         display_labels=['SP', 'NO SP'],
                                         cmap=plt.cm.Blues,
                                         normalize=normalize)
            disp.figure_.suptitle(title, fontsize=14, verticalalignment='center')
            disp.ax_.set_title(f"Model: {'MLP'}, Data: aminoacid {coding}", fontsize=10)

            plt.show()


mlp = MultiLayerPerception()
mlp.parse_file()

for coding in ['identities', 'characteristics']:
    print(coding)
    if coding == 'identities':
        mlp.pre_process_identities_data()
    mlp.split_data(coding)
    mlp.classifier()
    mlp.confusion_matrix(coding)
