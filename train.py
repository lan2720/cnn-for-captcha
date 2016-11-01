# coding: utf-8

import os
import time
import datetime
import tensorflow as tf
from data_helpers import Dataset
from captcha_cnn import CaptchaCNN

# Model Hyperparameters
tf.flags.DEFINE_float("init_learning_rate", 1e-5, "The initial learning rate")
# Misc Parameters
tf.flags.DEFINE_string("checkpoint", '', "Resume checkpoint")

FLAGS = tf.flags.FLAGS

dataset = Dataset()

with tf.Graph().as_default():
	sess = tf.Session()
	with sess.as_default():
		model = CaptchaCNN(img_width=96,
						   img_height=25,
						   num_of_labels=dataset.num_of_labels,
						   filter_sizes=[5, 5, 3],
						   num_filters=[32, 32, 32],
						   maxpool_sizes=[2, 2, 2])

		# Define Training procedure
		global_step = tf.Variable(0, name="global_step", trainable=False)
		# optimizer = tf.train.AdamOptimizer(1e-3, epsilon=1e-6)
		optimizer = tf.train.AdamOptimizer(FLAGS.init_learning_rate)
		grads_and_vars = optimizer.compute_gradients(model.reduced_loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

		# # Output directory for models and summaries
		# if FLAGS.checkpoint == "":
		# 	timestamp = str(int(time.time()))
		# 	out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
		# 	print("Writing to {}\n".format(out_dir))
		# else:
		# 	out_dir = FLAGS.checkpoint
		#
		# # Summaries for loss and accuracy
		# loss_summary = tf.scalar_summary("loss", model.reduced_loss)
		# acc_summary = tf.scalar_summary("accuracy", model.accuracy)
		#
		# # Train Summaries
		# train_summary_op = tf.merge_summary([loss_summary, acc_summary])
		# train_summary_dir = os.path.join(out_dir, "summaries", "train")
		# train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)
		#
		# # Dev summaries
		# dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
		# dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
		# dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

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

			_, step, scores, predictions, accuracy, loss = sess.run(
				[train_op, global_step, model.scores, model.predictions, model.accuracy, model.reduced_loss], feed_dict)
			time_str = datetime.datetime.now().strftime("%d, %b %Y %H:%M:%S")
			print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
			# train_summary_writer.add_summary(summaries, step)
		# print scores.shape
		# print predictions.shape
		# print accuracy


		# Generate batches
		train_batches = dataset.batch_generator()
		# Training loop. For each batch...
		for x_batch, y_batch in train_batches:
			train_step(x_batch, y_batch)
		# current_step = tf.train.global_step(sess, global_step)
		# if current_step % FLAGS.evaluate_every == 0:
		# 	print("\nEvaluation:")
		# 	test_step(dataset.x_test, dataset.y_test)
		# 	print("")
		# if current_step % FLAGS.checkpoint_every == 0:
		# 	path = saver.save(sess, checkpoint_prefix, global_step=current_step)
		# 	print("Saved model checkpoint to {}\n".format(path))
