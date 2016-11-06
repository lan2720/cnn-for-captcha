# coding=utf-8
"""
仿照tensorflow/models/image/cifar10/cifar10_input.py的处理方法生产验证码图片数据
"""

import os
import glob
import tensorflow as tf


def read_captcha(filename_queue):
	"""Reads and parses examples from Captcha data files.

	Recommendation: if you want N-way read parallelism, call this function
	N times.  This will give you N independent Readers reading different
	files & positions within those files, which will give better mixing of
	examples.

	Args:
		filename_queue: A queue of strings with the filenames to read from.

	Returns:
	An object representing a single example, with the following fields:
		height: number of rows in the result (32)
		width: number of columns in the result (32)
		depth: number of color channels in the result (3)
		key: a scalar string Tensor describing the filename & record number
			for this example.
		label: an int32 Tensor with the label in the range 0..9.
		uint8image: a [height, width, depth] uint8 Tensor with the image data
	"""

	class CaptchaRecord(object):
		pass

	result = CaptchaRecord()

	# Dimensions of the images in the CIFAR-10 dataset.
	# See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
	# input format.
	label_bytes = 1  # 2 for CIFAR-100
	result.height = 25
	result.width = 96
	result.depth = 1
	image_bytes = result.height * result.width * result.depth
	# Every record consists of a label followed by the image, with a
	# fixed number of bytes for each.
	record_bytes = label_bytes + image_bytes


def distorted_inputs(data_dir, batch_size, num_epochs=50, shuffle=True):
	"""Construct preprocessed input for Captcha training using the Reader ops.

	Args:
		data_dir: Path to the raw pics data directory.
		batch_size: Number of images per batch.

	Returns:
		images: Images. 4D tensor of [batch_size, height, width, 3] size.
		labels: Labels. 1D tensor of [batch_size, num_of_labels] size. (one hot)
	"""
	# filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
	#              for i in xrange(1, 6)]
	# for f in filenames:
	#     if not tf.gfile.Exists(f):
	#         raise ValueError('Failed to find file: ' + f)

	# Create a queue that produces the filenames to read.
	filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(os.path.join(data_dir, "*.jpg")),
	                                                num_epochs=num_epochs, shuffle=shuffle)

	# Read examples from files in the filename queue.
	read_input = read_captcha(filename_queue)
	reshaped_image = tf.cast(read_input.uint8image, tf.float32)

	height = IMAGE_SIZE
	width = IMAGE_SIZE

	# Image processing for training the network. Note the many random
	# distortions applied to the image.

	# Randomly crop a [height, width] section of the image.
	distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

	# Randomly flip the image horizontally.
	distorted_image = tf.image.random_flip_left_right(distorted_image)

	# Because these operations are not commutative, consider randomizing
	# the order their operation.
	distorted_image = tf.image.random_brightness(distorted_image,
	                                             max_delta=63)
	distorted_image = tf.image.random_contrast(distorted_image,
	                                           lower=0.2, upper=1.8)

	# Subtract off the mean and divide by the variance of the pixels.
	float_image = tf.image.per_image_whitening(distorted_image)

	# Ensure that the random shuffling has good mixing properties.
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
	                         min_fraction_of_examples_in_queue)
	print ('Filling queue with %d CIFAR images before starting to train. '
	       'This will take a few minutes.' % min_queue_examples)

	# Generate a batch of images and labels by building up a queue of examples.
	return _generate_image_and_label_batch(float_image, read_input.label,
	                                       min_queue_examples, batch_size,
	                                       shuffle=True)
