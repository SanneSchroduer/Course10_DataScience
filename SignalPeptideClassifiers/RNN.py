import collections
import matplotlib.pyplot as plt
import numpy as np
import logging
import preprocessData
import tensorflow as tf
from tensorflow.keras import layers


logger = logging.getLogger('classifier')
logging.info('Start processing data')
logger.setLevel(logging.DEBUG)

def main():
    instances_aa, instances_characteristics, class_ids = preprocessData.parse_file()
    one_hots = preprocessData.preprocessing(instances_aa)
    seq_train, seq_test, class_ids_train, class_ids_test = preprocessData.split_data(one_hots, class_ids)
    model(seq_train, seq_test, class_ids_train, class_ids_test)


def model(seq_train, seq_test, class_ids_train, class_ids_test):

    model = tf.keras.Sequential()
    # Add an Embedding layer expecting input vocab of size 1000, and
    # output embedding dimension of size 64.
    model.add(layers.Embedding(input_dim=1000, output_dim=64))

    # Add a LSTM layer with 128 internal units.
    model.add(layers.LSTM(128))

    # Add a Dense layer with 10 units.
    model.add(layers.Dense(10))

    model.summary()

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    model.fit(seq_train,
              class_ids_train,
              validation_data=(seq_test, class_ids_test),
              batch_size=64,
              epochs=2)

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