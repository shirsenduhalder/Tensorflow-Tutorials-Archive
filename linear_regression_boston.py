import os, sys
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from sklearn.datasets import load_boston

def read_infile():
	data = load_boston()
	features = np.array(data.data)
	target = np.array(data.target)

	return features, target

def feature_normalize(data):
	mu = np.mean(data, axis=0)
	std = np.mean(data, axis=0)

	return (data - mu)/std

def append_bias(features, target):
	n_samples = features.shape[0]
	n_features = features.shape[1]
	intercept_feature = np.ones((n_samples, 1))
	X = np.concatenate((features, intercept_feature), axis=1)
	X = np.reshape(X, (n_samples, n_features + 1))
	Y = np.reshape(target, (n_samples, 1))

	return X, Y


features, target = read_infile()
z_features = feature_normalize(features)
X_input, Y_input = append_bias(z_features, target)
num_features = X_input.shape[1]

X = tf.placeholder(tf.float32, (None, num_features))
Y = tf.placeholder(tf.float32, (None, 1))
w = tf.Variable(tf.random_normal((num_features, 1)), name='weights')

pred = tf.matmul(X, w)
error = pred - Y

init = tf.global_variables_initializer()
learning_rate = 0.01
num_epochs = 1500
cost_trace = []
cost = tf.reduce_mean(tf.square(error))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
	sess.run(init)
	for i in range(num_epochs):
		sys.stdout.write("\rEpochs: {}/{}".format(i + 1, num_epochs))
		sys.stdout.flush()
		sess.run(train_op, feed_dict={X:X_input, Y:Y_input})
		cost_trace.append(sess.run(cost, feed_dict={X:X_input, Y:Y_input}))
	error_ = sess.run(error, feed_dict={X:X_input, Y:Y_input})
	pred_ = sess.run(pred, feed_dict={X:X_input, Y:Y_input})

print("\nMSE final: {}".format(cost_trace[-1]))

import matplotlib.pyplot as plt 
plt.ion()
plt.plot(cost_trace)

fig, ax = plt.subplots()
plt.scatter(Y_input, pred_)
ax.set_xlabel('Actual House Price')
ax.set_ylabel('Predicted House Price')

plt.show()