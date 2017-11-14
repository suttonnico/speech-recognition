import numpy as np
import tensorflow as tf


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


#dataset = tf.data.Dataset.from_tensor_slices((features_train, labels_train))

learning_rate = 0.01
training_iteration = 30
batch_size = 200
display_step = 2

x = tf.placeholder("float", [None, 13])
y = tf.placeholder("float", [None, 61])

W = tf.Variable(tf.zeros([13, 61]))#,dtype=features_train.dtype))
b = tf.Variable(tf.zeros([61]))#,dtype=labels_train.dtype))

with tf.name_scope("Wx_b") as scope:

    model = tf.nn.softmax(tf.matmul(x, W) + b)

w_h = tf.summary.histogram("weights", W)
b_h = tf.summary.histogram("bias", b)

with tf.name_scope("cost_function") as scope:

    cost_function = -tf.reduce_sum(y*tf.log(model))

    tf.summary.scalar("cost_funcion", cost_function)

with tf.name_scope("train") as scope:

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

init = tf.global_variables_initializer()

merged_summary_op = tf.summary.merge_all()

dataset = DataSet(features_train,labels_train)

with tf.Session() as sess:
    sess.run(init)

#    summary_writer = tf.train.write_graph('./',graph_or_graph_def=sess.graph_def)
    features_placeholder = tf.placeholder(features_train.dtype, features_train.shape)
    labels_placeholder = tf.placeholder(labels_train.dtype, labels_train.shape)


    for iteration in range(training_iteration):
        avg_cost = 0
        [N, M] = np.shape(features_train)
        total_batch = (int(N/batch_size))
        for i in range(total_batch):
            batch_xs , batch_ys = dataset.next_batch(batch_size)
            sess.run(optimizer,feed_dict={x: batch_xs, y:batch_ys})
            avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y:batch_ys})

        summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y:batch_ys})
        #summary_writer.

        if iteration % display_step == 0:
            print("Iteration:",iteration,"cost=",avg_cost)
