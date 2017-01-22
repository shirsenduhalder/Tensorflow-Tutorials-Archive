import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n_visible = 784
n_hidden = 500
display_step = 1
num_epochs = 200
batch_size = 32
lr = tf.constant(0.001, tf.float32)

x = tf.placeholder(tf.float32, [None, n_visible], name="x")
W = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01), name='W')
b_h = tf.Variable(tf.zeros([1, n_hidden], tf.float32, name="b_h"))
b_v = tf.Variable(tf.zeros([1, n_visible], tf.float32, name="b_v"))

def sample(probs):
	return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

def gibbs_step(x_k):
	h_k = sample(tf.nn.sigmoid(tf.matmul(x_k, W) + b_h))
	x_k = sample(tf.nn.sigmoid(tf.matmul(h_k, tf.transpose(W)) + b_v))

	return x_k

def gibbs_sample(k, x_k):
	for i in range(k):
		x_out = gibbs_step(x_k)

	return x_out

x_s = gibbs_sample(2, x)
h_s = sample(tf.nn.sigmoid(tf.matmul(x_s, W) + b_h))

h = sample(tf.nn.sigmoid(tf.matmul(x, W) + b_h))
x_ = sample(tf.nn.sigmoid(tf.matmul(h, tf.transpose(W)) + b_v))

size_batch = tf.cast(tf.shape(x)[0], tf.float32)
W_add = tf.multiply(lr/size_batch, tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_s), h_s)))
bv_add = tf.multiply(lr/batch_size, tf.reduce_sum(tf.subtract(x, x_s), 0, True))
bh_add = tf.multiply(lr/batch_size, tf.reduce_sum(tf.subtract(h, h_s), 0, True))
updt = [W.assign_add(W_add), b_v.assign_add(bv_add), b_h.assign_add(bh_add)]

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	total_batch = int(mnist.train.num_examples/batch_size)

	for epoch in range(num_epochs):
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)

			batch_xs = (batch_xs > 0)*1
			_ = sess.run([updt], feed_dict={x:batch_xs})

		if(epoch % display_step == 0):
			print("Epoch: {}".format(epoch + 1))

	print("RBM training completed")
	out = sess.run(h, feed_dict={x:(mnist.test.images[:20] > 0)*1})
	label = mnist.test.labels[:20]

	plt.figure(1)
	for k in range(20):
		plt.subplot(4, 5, k+1)
		image = (mnist.test.images[k]>0)*1
		image = np.reshape(image, (28, 28))
		plt.imshow(image, cmap='gray')

	plt.figure(2)
	for k in range(20):
		plt.subplot(4, 5, k+1)
		image = sess.run(x_, feed_dict={h:np.reshape(out[k], (-1, n_hidden))})
		image = np.reshape(image, (28, 28))
		plt.imshow(image, cmap='gray')
		print(np.argmax(label[k]))

	sess.close()