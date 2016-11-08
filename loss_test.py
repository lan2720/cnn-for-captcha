# coding=utf-8
from __future__ import print_function

import numpy as np
from loss import loss_for_multi_labels
import tensorflow as tf


def full_connect(inputs, weights_shape, biases_shape):
    with tf.device('/cpu:0'):
        weights = tf.get_variable("weights",
                                  weights_shape,
                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        biases = tf.get_variable("biases",
                                 biases_shape,
                                 initializer=tf.constant_initializer(0.0))
    return tf.matmul(inputs, weights) + biases


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # x is 2d array. softmax by row
    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape([-1, 1])


def np_check(logits, targets, num_of_labels, num_of_classes):
    losses = []
    for i in range(num_of_labels):
        part_of_logits = logits[:, i * num_of_classes:(i + 1) * num_of_classes]
        part_of_outputs = softmax(part_of_logits)
        correct_indices = targets[:, i]
        correct = np.choose(correct_indices, part_of_outputs.T)
        reduce_loss = np.mean(-np.log(correct))
        losses.append(reduce_loss)
    return np.mean(losses)


def test_loss_for_multi_labels():
    num_of_labels = 4
    num_of_classes = 5
    targets = np.random.randint(low=0, high=5, size=[10, num_of_labels], dtype=np.int32)
    inputs = np.random.uniform(0.0, 5.0, size=[10, 96])
    inputs = inputs.astype(np.float32)
    outputs = full_connect(inputs,
                           weights_shape=[96, num_of_labels * num_of_classes],
                           biases_shape=[num_of_labels * num_of_classes])
    # 以上为logits
    loss = loss_for_multi_labels(outputs, targets, num_of_labels, num_of_classes)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        logits_, loss_ = sess.run([outputs, loss])
        # 不应该在这里做softmax，而应该单独对part_of_output做softmax
        print("logits_:", logits_)
        print("")
        print("targets:", targets)
        print("tensorflow loss:", loss_)
        print("numpy check:", np_check(logits_, targets, num_of_labels, num_of_classes))


if __name__ == '__main__':
    test_loss_for_multi_labels()
