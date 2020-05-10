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

instances_aa, instances_characteristics, class_ids = preprocessData.parse_file()
one_hots = preprocessData.preprocessing(instances_aa)

print(one_hots)

model = tf.keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128))

# Add a Dense layer with 10 units.
model.add(layers.Dense(10))

model.summary()