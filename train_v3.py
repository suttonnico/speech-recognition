import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import reader_v2


class DataSet(object):

  def __init__(self, features, labels, fake_data=False, one_hot=False):
    """Construct a DataSet. one_hot arg is used only if fake_data is true."""

    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert features.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (features.shape,
                                                 labels.shape))
      self._num_examples = features.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      #assert images.shape[3] == 1
      #images = images.reshape(images.shape[0],
      #                        images.shape[1] * images.shape[2])
      # Convert from [0, 255] -> [0.0, 1.0].
      features = features.astype(np.float32)
      features = np.multiply(features, 1.0 / np.max(features))
    self._features = features
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def features(self):
    return self._features

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data    No me interesa shufflear
      #perm = np.arange(self._num_examples)
      #np.random.shuffle(perm)
      #self._images = self._images[perm]
      #self._labels = self._labels[perm]

      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._features[start:end], self._labels[start:end]


features_train = np.load('features_train.npy')
labels_train = np.load('labels_train.npy')
features_test = np.load('features_test.npy')
labels_test = np.load('labels_test.npy')

dataset = DataSet(features_train,labels_train)
test_dataset = DataSet(features_test,labels_test)

[N, M] = np.shape(features_train)

time_steps = 10
num_input = 13
num_output = 61
num_hidden = 24

data = tf.placeholder(tf.float32, [None, time_steps, num_input])
target = tf.placeholder(tf.float32, [None, num_output])


cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)

val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

