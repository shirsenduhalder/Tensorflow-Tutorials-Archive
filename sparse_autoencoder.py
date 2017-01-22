import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 200
batch_size = 100
display_step = 1
examples_to_show = 10

n_hidden_1 = 32*32
n_input = 28*28

X = tf.placeholder(tf.float32, [None, n_input])

weights = {
	'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

biases = {
	'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'decoder_b1': tf.Variable(tf.random_normal([n_input]))
}

def encoder(x):
	layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))

	return layer1

def decoder(x):
	layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, tf.transpose(weights['encoder_h1'])), biases['decoder_b1']))

	return layer1

def log_func(x1, x2):
	return tf.multiply(x1, tf.log(tf.div(x1, x2)))

def KL_div(rho, rho_hat):
	inv_rho = tf.subtract(tf.constant(1.), rho)
	inv_rhohat = tf.subtract(tf.constant(1.), rho_hat)
	log_rho = log_func(rho, rho_hat) + log_func(inv_rho, inv_rhohat)

	return log_rho

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)
rho_hat = tf.reduce_mean(encoder_op, 1)

#reconstruction
y_pred = decoder_op
y_true = X

cost_m = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
cost_sparse = 0.001*tf.reduce_mean(KL_div(0.2, rho_hat))
cost_reg = 0.0001*(tf.nn.l2_loss(weights['decoder_h1']) + tf.nn.l2_loss(weights['encoder_h1']))

cost = tf.add(cost_reg, tf.add(cost_sparse, cost_m))

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
start_time = time.time()

with tf.Session() as sess:
	sess.run(init)
	total_batch = int(mnist.train.num_examples/batch_size)

	for epoch in range(training_epochs):
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			_, c = sess.run([optimizer, cost], feed_dict={X:batch_xs})

		if(epoch % display_step == 0):
			print("Epoch: {}, Cost: {}".format(epoch, c))

	print("Optimization finished")

	encode_decode = sess.run(y_pred, feed_dict={X:mnist.test.images[:10]})

	f, a = plt.subplots(2, 10, figsize=(10, 2))

	for i in range(10):
		a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
		a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))

	dec = sess.run(weights['decoder_h1'])
	enc = sess.run(weights['encoder_h1'])

end_time = time.time()
print("Time elapsed: {}".format(end_time - start_time))