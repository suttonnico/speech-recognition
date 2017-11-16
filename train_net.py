import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

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


LOGDIR = '/logs'

features_train = np.load('features_train.npy')
labels_train = np.load('labels_train.npy')
features_test = np.load('features_test.npy')
labels_test = np.load('labels_test.npy')

dataset = DataSet(features_train,labels_train)
test_dataset = DataSet(features_test,labels_test)

#dataset = tf.data.Dataset.from_tensor_slices((features_train, labels_train))

learning_rate = 1
training_iteration = 30
batch_size = 200
display_step = 200
training_steps = 10000


#model
num_input = 13
num_hidden = 200        #estos dos
timesteps = 1         #son de prueba
num_classes = 61
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

#W = tf.Variable(tf.zeros([13, 61]))#,dtype=features_train.dtype))
#b = tf.Variable(tf.zeros([61]))#,dtype=labels_train.dtype))


# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

 #Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = dataset.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 1000
    test_data, test_label = dataset.next_batch(test_len)
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

exit(23)

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
