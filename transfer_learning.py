import os, sys
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2, glob2
from tensorflow.contrib.slim.nets import vgg
from mlxtend.preprocessing import shuffle_arrays_unison

batch_size = 16
learning_rate = 0.01
width = 224
height = 224
checkpoints_dir = 'checkpoints/'
slim = tf.contrib.slim

all_images = glob2.glob("dogsvscats/train/**/*.jpg")
train_images, validation_images = train_test_split(all_images, train_size=0.8, test_size=0.2)
MEAN_VALUE = np.array([103.939, 116.779, 123.68])

def image_preprocess(img_path, width, height):
	img = cv2.imread(img_path)
	img = cv2.resize(img, (width, height))

	if img.ndim == 2:
		img = np.tile(np.expand_dims(img, axis=-1), (1, 1, 3))

	img = img - MEAN_VALUE
	
	return img

def data_gen_small(images, batch_size, width, height):
	while True:
		ix = np.random.choice(np.arange(len(images)), batch_size)
		imgs = []
		labels = []

		for i in ix:
			data_dir = ''
			if 'cat' in images[i].split('/')[-2]:
				labels.append(1)
			else:
				labels.append(0)
			array_img = image_preprocess(images[i], width, height)
			imgs.append(array_img)

		imgs = np.array(imgs)
		labels = np.reshape(np.array(labels), (batch_size, 1))

		yield imgs, labels


train_gen = data_gen_small(train_images, batch_size, width, height)
validation_gen = data_gen_small(validation_images, batch_size, width, height)

with tf.Graph().as_default():

	x = tf.placeholder(tf.float32, [None, width,height, 3])
	y = tf.placeholder(tf.float32, [None, 1])

	with slim.arg_scope(vgg.vgg_arg_scope()):
		logits, end_points = vgg.vgg_16(x, num_classes=1000, is_training=False)
		fc_7 = end_points['vgg_16/fc7']

	W1 = tf.Variable(tf.random_normal([4096, 1], mean=0.0, stddev=0.02, name='W1'))
	b1 = tf.Variable(tf.random_normal([1], mean=0.0, stddev=0.02), name='b1')

	fc_7 = tf.reshape(fc_7, [-1, W1.get_shape().as_list()[0]])
	logitx = tf.nn.bias_add(tf.matmul(fc_7, W1), b1)
	probx = tf.nn.sigmoid(logitx)


	cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logitx, labels=y))
	
	# we only learn W1 and b1 and hence include them in var list
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, var_list=[W1, b1])

	init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir,'vgg_16.ckpt'), slim.get_model_variables('vgg_16'))

	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)

		init_fn(sess)

		for i in range(1):
			for j in range(50):
				batch_x, batch_y = next(train_gen)
				val_x, val_y = next(validation_gen)
				sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})
				cost_train = sess.run(cost,feed_dict={x:batch_x, y:batch_y})
				cost_val = sess.run(cost, feed_dict={x:val_x, y:val_y})
				prob_out = sess.run(probx, feed_dict={x:val_x, y:val_y})
				print("Batch: {}, Training cost: {}, Validation cost: {}".format(j, cost_train, cost_val))

		out_val = (prob_out > 0.5)*1
		print("Accuracy: {}".format(np.sum(out_val == val_y)*100/float(len(val_y))))