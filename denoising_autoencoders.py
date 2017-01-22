import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from skimage import transform
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt

def autoencoder(inputs):
	#encoder
	net = layers.conv2d(inputs, 32, [5, 5], stride=2, padding='SAME')
	net = layers.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
	net = layers.conv2d(net, 8, [5, 5], stride=4, padding='SAME')

	#decoder
	net = layers.conv2d_transpose(net, 16, [5, 5], stride=4, padding='SAME')
	net = layers.conv2d_transpose(net, 32, [5, 5], stride=2, padding='SAME')
	net = layers.conv2d_transpose(net, 1, [5, 5], stride=2, padding='SAME')

	return net

def resize_batch(imgs):
	imgs = imgs.reshape((-1, 28, 28, 1))
	resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))

	for i in range(imgs.shape[0]):
		resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))

	return resized_imgs

def gaussian_noisy(image):
	row, col = image.shape
	mean = 0
	var = 0.1
	sigma = var**0.5
	gauss = np.random.normal(mean, sigma, (row, col))
	gauss = gauss.reshape((row, col))
	noisy = image + gauss

	return noisy

def salt_pepper(image):
	row, col = image.shape
	s_vs_p = 0.5
	amount = 0.05
	out = np.copy(image)

	#salt
	num_salt = np.ceil(amount * image.size * s_vs_p)
	coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
	out[coords] = 1

	#pepper
	num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
	coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
	out[coords] = 0

	return out

a_e_inputs = tf.placeholder(tf.float32, (None, 32, 32, 1))
a_e_input_noise = tf.placeholder(tf.float32, (None, 32, 32, 1))
a_e_outputs = autoencoder(a_e_input_noise)

loss = tf.reduce_mean(tf.square(a_e_outputs - a_e_inputs))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

init = tf.global_variables_initializer()

batch_size = 100
epoch_num = 10
lr = 0.001

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_per_ep = mnist.train.num_examples//batch_size

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(epoch_num):
		for batch_num in range(batch_per_ep):
			batch_img, batch_label = mnist.train.next_batch(batch_size)
			batch_img = batch_img.reshape((-1, 28, 28, 1))
			batch_img = resize_batch(batch_img)

			image_arr = []
			for i in range(len(batch_img)):
				img = batch_img[i, ..., 0]
				img = gaussian_noisy(img)
				image_arr.append(img)

			image_arr = np.array(image_arr)
			image_arr = np.reshape(image_arr, (-1, 32, 32, 1))
			_, c = sess.run([train_op, loss], feed_dict={a_e_input_noise:image_arr, a_e_inputs:batch_img})
			print("Epoch: {}, Cost: {}".format(epoch, c))

	batch_img, batch_label = mnist.test.next_batch(50)
	batch_img = resize_batch(batch_img)
	image_arr = []

	for i in range(batch_img.shape[0]):
		img = batch_img[i, ..., 0]
		img = gaussian_noisy(img)
		image_arr.append(img)
	image_arr = np.array(image_arr)
	image_arr = np.reshape(image_arr, (-1, 32, 32, 1))

	reconst_img = sess.run([a_e_outputs], feed_dict={a_e_input_noise:image_arr})[0]

	plt.figure(1)
	plt.title("Input noisy images")

	for i in range(50):
		plt.subplot(5, 10, (i + 1))
		plt.imshow(image_arr[i, ..., 0], cmap='gray')

	plt.figure(2)
	plt.title("Reconstructed images")

	for i in range(50):
		plt.subplot(5, 10, (i + 1))
		plt.imshow(reconst_img[i, ..., 0], cmap='gray')

	plt.show()