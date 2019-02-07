from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

def multi_hot_sequences(sequences, dimension):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s
    return results


train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)
# plt.plot(train_data[0])
# plt.show()
#Build baseline model
baseline_model = keras.Sequential([
    # `input_shape` is only required here so that `.summary` works. 
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

baseline_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])

baseline_model.summary()

baseline_history = baseline_model.fit(train_data,
                                      train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2)
#END BASELINE MODEL

#Create smaller model to show underfitting
#uncomment to demonstrate underfitting
# smaller_model = keras.Sequential([
#     keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
#     keras.layers.Dense(4, activation=tf.nn.relu),
#     keras.layers.Dense(1, activation=tf.nn.sigmoid)
# ])

# smaller_model.compile(optimizer='adam',
#                 loss='binary_crossentropy',
#                 metrics=['accuracy', 'binary_crossentropy'])

# smaller_model.summary()

# smaller_history = smaller_model.fit(train_data,
#                                     train_labels,
#                                     epochs=20,
#                                     batch_size=512,
#                                     validation_data=(test_data, test_labels),
#                                     verbose=2)
#END SMALL MODEL

#Create large model to demonstrate overfitting uncomment to use
#Takes awhile
# bigger_model = keras.models.Sequential([
#     keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
#     keras.layers.Dense(512, activation=tf.nn.relu),
#     keras.layers.Dense(1, activation=tf.nn.sigmoid)
# ])

# bigger_model.compile(optimizer='adam',
#                      loss='binary_crossentropy',
#                      metrics=['accuracy','binary_crossentropy'])

# bigger_model.summary()

# bigger_history = bigger_model.fit(train_data, train_labels,
#                                   epochs=20,
#                                   batch_size=512,
#                                   validation_data=(test_data, test_labels),
#                                   verbose=2)
#END BIGGER MODEL

#L2 model uncomment to demonstrate using L2 as a means to solve this problem
#uncomment to test
# l2_model = keras.models.Sequential([
#     keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
#                        activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
#     keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
#                        activation=tf.nn.relu),
#     keras.layers.Dense(1, activation=tf.nn.sigmoid)
# ])

# l2_model.compile(optimizer='adam',
#                  loss='binary_crossentropy',
#                  metrics=['accuracy', 'binary_crossentropy'])
# l2_model.summary()
# l2_model_history = l2_model.fit(train_data, train_labels,
#                                 epochs=20,
#                                 batch_size=512,
#                                 validation_data=(test_data, test_labels),
#                                 verbose=2)
#END L2 MODEL

#Dropout model
dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

dpt_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy','binary_crossentropy'])
dpt_model.summary()
dpt_model_history = dpt_model.fit(train_data, train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)

#Plot training and vailidation loss
def plot_history(histories, key='binary_crossentropy'):
  plt.figure(figsize=(16,10))
    
  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])

#uncomment to see graph example of each model compared to training data
# plot_history([('baseline', baseline_history),
#               ('smaller', smaller_history),
#               ('bigger', bigger_history)])
#uncomment to see graph example of l2 model against baseline model to training data
# plot_history([('baseline', baseline_history),
#               ('l2', l2_model_history)])
#compares dropout model to baseline against training data
plot_history([('baseline', baseline_history),
              ('dropout', dpt_model_history)])
#display graph
plt.show()