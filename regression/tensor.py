from __future__ import absolute_import, division, print_function



import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#download data for cars
dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
#format data
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin'] 
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)
#Display data/format data
dataset = raw_dataset.copy()
print(dataset.tail())
dataset = dataset.dropna()
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
print(dataset.tail())

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
#Show graph displaying different labels of the data and how they differ
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)
#Separate the target value, or "label", from the features. 
# This label is the value that will train the model to predict.
bias_initializer = tf.constant_initializer(value=1.5)
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')
#Normalize the data
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
#Build the model
def build_model():
  model = keras.Sequential([
    #UPDATE made model layers more dense as with lack of training data
    # Approaching problem with process of making a large input pool to be simplified for stronger training
    layers.Dense(32, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(16, activation=tf.nn.relu),
    # layers.Dense(8, activation=tf.nn.relu),
    # layers.Dense(16, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

#Instantiate model
model = build_model()
# The patience parameter is the amount of epochs to check for improvement
#10 has been chosen to give the model some learning room to avoid overfitting.
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


#Describe the model
model.summary()
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
# print(example_result)
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())
#create graph of training data
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,5])
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,20])
plot_history(history)
# Evaluate model uncomment below
test_predictions = model.predict(normed_test_data).flatten()
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
plt.margins(x=0, y=-0.25)
plt.plot([-100, 100], [-100, 100])
#Histogram graph 
# error = test_predictions - test_labels
# plt.hist(error, bins = 25)
# plt.xlabel("Prediction Error [MPG]")
# plt.ylabel("Count")
#Display graphs
plt.show()
