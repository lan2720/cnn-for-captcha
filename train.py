# coding=utf-8
from __future__ import print_function

import time
import datetime
from configs import *
import tensorflow as tf
from captcha_cnn import CaptchaCNN
from prediction import prediction_for_multi_labels
from loss import loss_for_multi_labels
from metrics import accuracy_for_multi_labels
from captcha_input import generate_datasets_tfrecords, input_pipeline

# Define parameters
flags = tf.app.flags
FLAGS = flags.FLAGS
# Model Hyperparameters
tf.flags.DEFINE_float("init_learning_rate", 1e-3, "The initial learning rate")
# Misc Parameters
tf.flags.DEFINE_string("checkpoint_dir", '', "Indicates the checkpoint directory")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model every certain steps (default: 100)")
tf.flags.DEFINE_integer("save_every", 5000, "Save model every certain steps (default: 1000)")
tf.flags.DEFINE_string("optimizer", "momentum", "The training algorithm to use")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("mode", "train_from_scratch", "Option mode: train, train_from_scratch, inference")

print("Loading dataset...")
# 如果存在tfrecords直接使用，如果要重新生成，则先删除tfrecords文件夹，再运行
if not os.path.exists(os.path.join(os.path.dirname(__file__), "tfrecords")):
    generate_datasets_tfrecords(DATA_BATCHES_DIR, one_hot=False)

# Create session to run graph
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Train batch data
        train_images_batch, train_labels_batch = input_pipeline(one_hot=False, batch_size=60, num_epochs=200,
                                                                name='train')
        # Validation batch data
        # valid_images_batch, valid_labels_batch = input_pipeline(one_hot=False, batch_size=50, num_epochs=1,
        #                                                         name='validation')
        # Define model
        captchaCNN = CaptchaCNN(filter_sizes=[5, 5, 3, 3],
                                num_of_filters=[32, 32, 32, 32],
                                filter_strides=[1, 1, 1, 1],
                                pool_sizes=[2, 2, 2, 2],
                                pool_strides=[1, 1, 1, 1],
                                pool_types=['max', 'avg', 'avg', 'avg'],
                                input_channels=NUM_CHANNELS,
                                hidden_sizes=[256],
                                num_of_labels=NUM_OF_LABELS,
                                num_of_classes=NUM_OF_CLASSES
                                )

        # Define model
        captchaCNN.flow(train_images_batch)

        # Define loss
        loss = loss_for_multi_labels(captchaCNN.outputs, train_labels_batch, NUM_OF_LABELS, NUM_OF_CLASSES)

        # Define predictions
        predictions = prediction_for_multi_labels(captchaCNN.outputs, NUM_OF_LABELS, NUM_OF_CLASSES)

        # Define accuracy
        accuracy = accuracy_for_multi_labels(predictions, train_labels_batch)

        # learning rate decay
        with tf.device("/cpu:0"):
            global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_rate = tf.train.exponential_decay(FLAGS.init_learning_rate, global_step,
                                                   3000, 0.96, staircase=True)

        # Define optimizer
        print("Use the optimizer: {}".format(FLAGS.optimizer))
        if FLAGS.optimizer == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif FLAGS.optimizer == "momentum":
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True)
        elif FLAGS.optimizer == "adadelta":
            optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        elif FLAGS.optimizer == "adagrad":
            optimizer = tf.train.AdagradOptimizer(learning_rate)
        elif FLAGS.optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif FLAGS.optimizer == "ftrl":
            optimizer = tf.train.FtrlOptimizer(learning_rate)
        elif FLAGS.optimizer == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
        else:
            raise ValueError("Unknown optimizer: {}".format(FLAGS.optimizer))

        # Define train_op
        train_op = optimizer.minimize(loss, global_step=global_step)

        # Add an op to initialize the variables
        init_op = tf.initialize_all_variables()

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # 模式选择
        if FLAGS.mode == "train_from_scratch" or FLAGS.mode == "train":
            if FLAGS.mode == "train_from_scratch":
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                print("Train from scratch. Writing to {}\n".format(out_dir))
            else:
                if not FLAGS.checkpoint_dir:  # runs/146334553996
                    raise ValueError(
                        "Now mode is `{}`. Please give a checkpoint dir. (e.g: runs/1476415419)".format(FLAGS.mode))
                out_dir = FLAGS.checkpoint_dir
                ckpt = tf.train.get_checkpoint_state(os.path.join(out_dir, "checkpoints"))
                if ckpt and ckpt.model_checkpoint_path:
                    print("Continue training from the model {}".format(
                        ckpt.model_checkpoint_path))
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    raise ValueError("No valid checkpoint model in {}".format(FLAGS.checkpoint_dir))

            # 只有train和train_from_scratch需要summary
            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", loss)
            acc_summary = tf.scalar_summary("accuracy", accuracy)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

            sess.run(init_op)
            sess.run(tf.initialize_local_variables())

            # Get coordinator and run queues to read data
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            # print(train_images_batch[0].eval())

            try:
                while not coord.should_stop():
                    _, train_summaries, loss_value, preds_, acc, step = sess.run(
                        [train_op, train_summary_op, loss, predictions, accuracy, global_step])
                    time_str = datetime.datetime.now().strftime("%d, %b %Y %H:%M:%S")
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss_value, acc))
                    print("predictions[:10]:\n", preds_[:10])
                    # if step % FLAGS.evaluate_every == 0:
                    #     accuracy_value, auc_value, summary_value = sess.run(
                    #         [accuracy, summary_op])
                    #     end_time = datetime.datetime.now()
                    #     print("[{}] Step: {}, loss: {}, accuracy: {}, auc: {}".format(
                    #         end_time - start_time, step, loss_value, accuracy_value,
                    #         auc_value))
                    train_summary_writer.add_summary(train_summaries, step)

                    if step % FLAGS.save_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=step)
                        print("Saved model to {}\n".format(path))

            except tf.errors.OutOfRangeError:
                print("Done training after reading all data")
            finally:
                coord.request_stop()

            # Wait for threads to exit
            coord.join(threads)

        # elif FLAGS.mode == "inference":
        #     print("Start to run inference")
        #     if not FLAGS.checkpoint_dir:
        #         raise ValueError(
        #             "Now mode is `{}`. Please give a checkpoint dir. (e.g: runs/1476415419)".format(FLAGS.mode))
        #     start_time = datetime.datetime.now()
        #     inference_data = xxxx
        #
        #     # Restore weights from model file
        #     ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        #     if ckpt and ckpt.model_checkpoint_path:
        #         print("Use the model {}".format(ckpt.model_checkpoint_path))
        #         saver.restore(sess, ckpt.model_checkpoint_path)
        #         # todo: 在model中定义inference这个variable_scope
        #         inference_result = sess.run(inference_op, feed_dict={inference_images: inference_data})
        else:
            pass
