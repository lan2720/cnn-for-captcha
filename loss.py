# coding=utf-8

import tensorflow as tf


def loss_for_multi_labels(outputs, targets, num_of_labels, num_of_classes):
    """
    注意targets不是one-hot了，应该类似于[[2,3,1,0], [1,3,2,2], ...]
    以上样例为：batch_size个examples，每个example有num_of_labels个labels，每个值在[0, num_of_classes)之间
    """
    losses = []
    for i in range(num_of_labels):
        part_of_outputs = tf.slice(outputs,
                                   begin=[0, i * num_of_classes],
                                   size=[-1, num_of_classes],
                                   name="part-outputs-{}".format(i + 1))
        part_of_targets = tf.slice(targets,
                                   begin=[0, i],
                                   size=[-1, 1],
                                   name="part-targets-{}".format(i + 1))
        part_of_targets = tf.reshape(part_of_targets, [-1])
        part_of_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(part_of_outputs,
                                                                      part_of_targets,
                                                                      name="part-loss-{}".format(i + 1))
        reduced_part_of_loss = tf.reduce_mean(part_of_loss)
        losses.append(reduced_part_of_loss)
    loss = tf.reduce_mean(losses, name="loss")
    return loss
