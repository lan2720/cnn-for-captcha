# coding=utf-8

"""
Implement some metrics component, such as `accuracy`, `auc`, etc.
"""

import tensorflow as tf


def accuracy_for_multi_labels(predictions, targets):
    # The one difference between accuracy for exclusive and multi-label is:
    # predictions should be element-wise equally to targets, no argmax() equal
    # predictions and targets' shape = [batch_size, num_of_labels] (no one-hot)
    correct_prediction = tf.cast(tf.equal(predictions, targets), tf.float32)
    correct_prediction = tf.reduce_mean(correct_prediction, reduction_indices=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(correct_prediction, 1.0), tf.float32))
    return accuracy
