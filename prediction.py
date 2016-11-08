# coding=utf-8

import tensorflow as tf


def prediction_for_multi_labels(outputs, num_of_labels, num_of_classes):
    predictions = []
    for i in range(num_of_labels):
        # part_of_outputs shape = [batch_size, num_of_classes]
        part_of_outputs = tf.slice(outputs,
                                   begin=[0, i * num_of_classes],
                                   size=[-1, num_of_classes],
                                   name="part-outputs-{}".format(i + 1))
        # part_of_prediction shape = [batch_size, ]
        part_of_prediction = tf.argmax(part_of_outputs, dimension=1, name="part-predictions-{}".format(i + 1))
        part_of_prediction = tf.cast(part_of_prediction, tf.int32)
        predictions.append(part_of_prediction)
    # pack -> [batch_size, num_of_labels]
    predictions = tf.pack(predictions, axis=1, name="joint-predictions")
    return predictions
