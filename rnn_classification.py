import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 50
num_train = mnist.train.num_examples
num_batches = (num_train//batch_size) + 1
epochs = 2

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

def RNN(x, weights, biases):

	x = tf.unstack(x, n_steps, axis=1)
	lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0)

	outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)
	return (tf.matmul(outputs[-1], weights['out']) + biases['out'])

x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
	'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}

biases = {
	'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = RNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	i=0

	while i<epochs:
		for step in range(num_batches):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			batch_x = batch_x.reshape((batch_size, n_steps, n_input))
			sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})
			if((step + 1 ) % display_step == 0):
				acc = sess.run(accuracy, feed_dict={x:batch_x, y:batch_y})
				loss = sess.run(cost, feed_dict={x:batch_x, y:batch_y})
				print("Epoch: {}, Step: {}, Loss: {}, Accuracy: {}".format(i, step, loss, acc))
		i+=1
	print("Optimization Completed")

	test_len = 500
	test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
	test_labels = mnist.test.labels[:test_len]
	print("TEST Accuracy: {}".format(sess.run(accuracy, feed_dict={x:test_data, y:test_labels})))