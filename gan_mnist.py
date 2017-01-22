import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
h1_dim = 150
h2_dim = 300
dim = 100
batch_size = 64

def generator(z_noise):
	w1 = tf.Variable(tf.truncated_normal([dim, h1_dim], stddev=0.1), name="w1_g", dtype=tf.float32)
	b1 = tf.Variable(tf.zeros([h1_dim]), name="b1_g", dtype=tf.float32)
	h1 = tf.nn.relu(tf.matmul(z_noise, w1) + b1)

	w2 = tf.Variable(tf.truncated_normal([h1_dim, h2_dim], stddev=0.1), name="w2_g")
	b2 = tf.Variable(tf.zeros([h2_dim]), name="b2_g", dtype=tf.float32)
	h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

	w3 = tf.Variable(tf.truncated_normal([h2_dim, 28*28], stddev=0.1), name="w2_g")
	b3 = tf.Variable(tf.zeros([28*28]), name="b3_g", dtype=tf.float32)
	h3 = tf.matmul(h2, w3) + b3

	out_gen = tf.nn.tanh(h3)

	weights_g = [w1, b1, w2, b2, w3, b3]

	return out_gen, weights_g

def discriminator(x, out_gen, keep_prob):
	x_all = tf.concat([x, out_gen], 0)
	
	w1 = tf.Variable(tf.truncated_normal([28*28, h2_dim], stddev=0.1), name="w1_d", dtype=tf.float32)
	b1 = tf.Variable(tf.zeros([h2_dim]), name="b1_d", dtype=tf.float32)
	h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_all, w1) + b1), keep_prob)

	w2 = tf.Variable(tf.truncated_normal([h2_dim, h1_dim], stddev=0.1), name="w2_d", dtype=tf.float32)
	b2 = tf.Variable(tf.zeros([h1_dim]), name="b2_d", dtype=tf.float32)
	h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob)

	w3 = tf.Variable(tf.truncated_normal([h1_dim, 1], stddev=0.1), name="w3_d", dtype=tf.float32)
	b3 = tf.Variable(tf.zeros([1]), name="b3_d", dtype=tf.float32)
	h3 = tf.matmul(h2, w3) + b3

	y_data = tf.nn.sigmoid(tf.slice(h3, [0, 0], [batch_size, -1]))
	y_fake = tf.nn.sigmoid(tf.slice(h3, [batch_size, 0], [-1, -1]))

	weights_d = [w1, b1, w2, b2, w3, b3]

	return y_data, y_fake, weights_d

x = tf.placeholder(tf.float32, [batch_size, 28*28], name="x_data")
z_noise = tf.placeholder(tf.float32, [batch_size, dim], name='z_prior')

keep_prob = tf.placeholder(tf.float32, name='keep_prob')

out_gen, weights_g = generator(z_noise)
y_data, y_fake, weights_d = discriminator(x, out_gen, keep_prob)

discr_loss = -1*tf.reduce_mean(tf.log(y_data) + tf.log(1 - y_fake))
gen_loss = -1* tf.reduce_mean(tf.log(y_fake))
optimzier = tf.train.AdamOptimizer(0.0001)
d_trainer = optimzier.minimize(discr_loss, var_list=weights_d)
g_trainer = optimzier.minimize(gen_loss, var_list=weights_g)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

sess = tf.Session()
sess.run(init)

z_sample = np.random.uniform(-1, 1, size=(batch_size, dim)).astype(np.float32)

for i in range(60000):
	batch_x, _ = mnist.train.next_batch(batch_size)
	x_value = 2*batch_x.astype(np.float32) - 1
	z_value = np.random.uniform(-1, 1, size=(batch_size, dim)).astype(np.float32)
	sess.run(d_trainer, feed_dict={x:x_value, z_noise:z_value, keep_prob:0.7})
	sess.run(g_trainer, feed_dict={x:x_value, z_noise:z_value, keep_prob:0.7})
	[c1, c2] = sess.run([discr_loss, gen_loss], feed_dict={x:x_value, z_noise:z_value, keep_prob:0.7})
	print("Iteration: {}, Disc Loss:{}, Gen Loss:{}".format(i, c1, c2))

out_val_img = sess.run(out_gen, feed_dict={z_noise:z_sample})
saver.save(sess, "checkpoints/newgan1",global_step=1)

imgs = 0.5*(out_val_img + 1)

for k in range(36):
	plt.subplot(6, 6, (k + 1))
	image = np.reshape(imgs[k], (28, 28))
	plt.imshow(image, cmap='gray')