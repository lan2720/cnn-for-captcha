# coding: utf-8

import os
import time
import datetime
import tensorflow as tf
import numpy as np
from data_helpers import Dataset
from captcha_cnn import CaptchaCNN

# Model Hyperparameters
tf.flags.DEFINE_float("init_learning_rate", 1e-4, "The initial learning rate")
# Misc Parameters
tf.flags.DEFINE_string("checkpoint", '', "Resume checkpoint")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model every certain steps (default: 100)")

tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

print("Loading dataset...")
dataset = Dataset(extfile="150000.npz")
X = dataset.images  # (149998, 25, 96, 3)
y = dataset.labels  # (149998, 258)
X_train, X_test = X[:50000], X[-100:]
y_train, y_test = y[:50000], y[-100:]
# normalize( only compute mean and std on trainset)
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train = (X_train - X_mean) / (X_std + 0.00001)
X_test = (X_test - X_mean) / (X_std + 0.00001)




with tf.Graph().as_default():
	session_conf = tf.ConfigProto(
		log_device_placement=FLAGS.log_device_placement)
	sess = tf.Session(config=session_conf)
	with sess.as_default():
		model = CaptchaCNN(img_width=96,
						   img_height=25,
						   num_of_chars=6,
						   num_of_labels=dataset.num_of_labels,
						   input_channels=3,
						   filter_sizes=[2, 2, 2],
						   num_filters=[32, 64, 64],
						   maxpool_sizes=[2, 2, 2],
						   hidden_size=1024)

		# Define Training procedure
		global_step = tf.Variable(0, name="global_step", trainable=False)
		# optimizer = tf.train.AdamOptimizer(1e-3, epsilon=1e-6)
		optimizer = tf.train.AdamOptimizer(FLAGS.init_learning_rate)
		grads_and_vars = optimizer.compute_gradients(model.loss)
		# print "trainable variables:"
		# print tf.trainable_variables()
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

		# Output directory for models and summaries
		if FLAGS.checkpoint == "":
			timestamp = str(int(time.time()))
			out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
			print("Writing to {}\n".format(out_dir))
		else:
			out_dir = FLAGS.checkpoint

		# Summaries for loss and accuracy
		loss_summary = tf.scalar_summary("loss", model.loss)
		acc_summary = tf.scalar_summary("accuracy", model.accuracy)

		# Train Summaries
		train_summary_op = tf.merge_summary([loss_summary, acc_summary])
		train_summary_dir = os.path.join(out_dir, "summaries", "train")
		train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

		# Dev summaries
		dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
		dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
		dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

		# Initialize all variables
		sess.run(tf.initialize_all_variables())


		def train_step(x_batch, y_batch):
			"""
			A single training step
			"""
			feed_dict = {
				model.images: x_batch,
				model.labels: y_batch
			}

			_, step, summaries, accuracy, individual_accuracy, loss, predictions = sess.run(
				[train_op, global_step, train_summary_op, model.accuracy, model.individual_accuracy, model.loss,
				 model.predictions], feed_dict)
			time_str = datetime.datetime.now().strftime("%d, %b %Y %H:%M:%S")

			# for pred, truth in zip(predictions.tolist(), y_batch.tolist())[:10]:
			# 	ls = []
			# 	for i in range(6):
			# 		index = np.argmax(np.asarray(truth[i * dataset.num_of_labels:(i + 1) * dataset.num_of_labels]))
			# 		ls.append(index)
			# 	print(
			# 		"{}  {}".format("".join([dataset.chars[i] for i in pred]),
			# 						"".join([dataset.chars[i] for i in ls])))

			print("{}: step {}, loss {:g}, individual_acc {:g}, acc {:g}".format(time_str, step, loss,
																				 individual_accuracy, accuracy))

			train_summary_writer.add_summary(summaries, step)


		def test_step():
			for x_batch, y_batch in dataset.batch_generator(X_test, y_test, batch_size=100, num_of_epoches=1,
															shuffle=False):
				feed_dict = {
					model.images: x_batch,
					model.labels: y_batch
				}
				step, loss, accuracy, individual_accuracy = sess.run(
					[global_step, model.loss, model.accuracy, model.individual_accuracy], feed_dict)
				time_str = datetime.datetime.now().strftime("%d, %b %Y %H:%M:%S")
				print("{}: step {}, loss {:g}, individual_acc {:g}, acc {:g}".format(time_str, step, loss,
																					 individual_accuracy, accuracy))


		# Training loop. For each batch...
		for x_batch, y_batch in dataset.batch_generator(X_train, y_train, batch_size=64, num_of_epoches=200,
														shuffle=True):
			train_step(x_batch, y_batch)
			current_step = tf.train.global_step(sess, global_step)
			if current_step % FLAGS.evaluate_every == 0:
				print("\nEvaluation:")
				test_step()
				print("")

			# if current_step % FLAGS.checkpoint_every == 0:
			# 	path = saver.save(sess, checkpoint_prefix, global_step=current_step)
			# 	print("Saved model checkpoint to {}\n".format(path))
