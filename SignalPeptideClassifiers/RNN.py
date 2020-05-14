import collections
import matplotlib.pyplot as plt
import numpy as np
import logging
import preprocessData
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers


logger = logging.getLogger('classifier')
logging.info('Start processing data')
logger.setLevel(logging.DEBUG)

def main():
    # instances_aa, instances_characteristics, class_ids = preprocessData.parse_file()
    # one_hots = preprocessData.preprocessing(instances_aa)
    # seq_train, seq_test, class_ids_train, class_ids_test = preprocessData.split_data(one_hots, class_ids)
    # model(instances_aa, seq_train, seq_test, class_ids_train, class_ids_test)
    test()

def model(instances_aa, seq_train, seq_test, class_ids_train, class_ids_test):


    print(instances_aa[3])

    instances_aa = ''.join(instances_aa[0])
    print(instances_aa)
    encoder = instances_aa.encoder

    print('Vocabulary size: {}'.format(encoder.vocab_size))

    #sample_string = 'Hello TensorFlow.'

    # encoded_string = instances_aa[3].encode(sample_string)
    # print('Encoded string is {}'.format(encoded_string))

    # original_string = instances_aa[3].decode(encoded_string)
    # print('The original string: "{}"'.format(original_string))

    # model = tf.keras.Sequential()
    # # Add an Embedding layer expecting input vocab of size 1000, and
    # # output embedding dimension of size 64.
    # model.add(layers.Embedding(input_dim=1000, output_dim=64))
    #
    # # Add a LSTM layer with 128 internal units.
    # model.add(layers.LSTM(128))
    #
    # # Add a Dense layer with 10 units.
    # model.add(layers.Dense(10))
    #
    # model.summary()
    #
    # array_seq_train = np.array(seq_train)
    # print(array_seq_train[:10])
    #
    # array_class_ids_train = np.array(class_ids_train)
    # print(array_class_ids_train[:10])
    #
    # model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    # print(class_ids_train[:10])
    # model.fit(seq_train,
    #           class_ids_train
    #           #validation_data=(seq_test, class_ids_test),
    #           #batch_size=64,
    #           #epochs=2
    #           )



def test():
    dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                              as_supervised=True)
    train_examples, test_examples = dataset['train'], dataset['test']
    encoder = info.features['text'].encoder


    # print('Vocabulary size: {}'.format(encoder.vocab_size))
    #
    # sample_string = 'Hello TensorFlow.'
    #
    # encoded_string = encoder.encode(sample_string)
    # print('Encoded string is {}'.format(encoded_string))
    #
    # original_string = encoder.decode(encoded_string)
    # print('The original string: "{}"'.format(original_string))
main()

"""
model = tf.keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128))

# Add a Dense layer with 10 units.
model.add(layers.Dense(10))

model.summary()
"""

#If sequence matters you need to use the first way so your input shape will be (batch_size, sequence_size, num_labels)