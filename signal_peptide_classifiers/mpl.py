import preprocess_data
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.neural_network import MLPClassifier

instances_aa, instances_characteristics, class_ids = preprocess_data.parse_file()
one_hots = preprocess_data.preprocessing(instances_aa)
seq_train, seq_test, class_ids_train, class_ids_test = preprocess_data.split_data(one_hots, class_ids)

#print(seq_train)
# print(class_ids_train)
#print(instances_aa)

X = [[0., 0.], [1., 1.]]
y = [0, 1]
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

classifier = clf.fit(seq_train, class_ids_train)

titles_options = [("Confusion matrix, without normalization", None), ("Normalized confusion matrix", 'true')]

for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, seq_test, class_ids_test,
                                 display_labels=['SP', 'NO SP'],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.figure_.suptitle(title, fontsize=14, verticalalignment='center')
    disp.ax_.set_title(f"Model: {'MLP'}, Data: aminoacid {'identities'}", fontsize=10)

    print(title)
    print(disp.confusion_matrix)

    plt.show()