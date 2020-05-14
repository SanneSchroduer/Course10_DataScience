import logging
import re
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger('classifier')
logging.info('Start processing data')
logger.setLevel(logging.DEBUG)

# source: http://www.chem.ucalgary.ca/courses/351/Carey5th/Ch27/ch27-1-4-2.html
# dictionary with pI values per amino acid
aa_pI = {"G": 5.97, "A": 6.00, "V": 5.96, "L": 5.98, "I": 6.02, "M": 5.74,
         "P": 6.30, "F": 5.48, "W": 5.89, "N": 5.41, "Q": 5.65, "S": 5.68, "T": 5.60,
         "Y": 5.66, "C": 5.07, "D": 2.77, "E": 3.22, "K": 9.74, "R": 10.76, "H": 7.59}

def parse_file():
    train_file = open('../data/train_set.fasta').readlines()
    test_file = open('../data/benchmark_set.fasta').readlines()
    logger.info(' Read training file')

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
    seq_train, seq_test, class_ids_train, class_ids_test = train_test_split(data, class_ids, test_size=0.20, random_state=42)

    return seq_train, seq_test, class_ids_train, class_ids_test