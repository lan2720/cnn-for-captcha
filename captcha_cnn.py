# coding: utf-8

import numpy as np
import tensorflow as tf


# filter_sizes = [5, 5, 3]
# num_filters = [32, 32, 32]
# maxpool_sizes = [2, 2, 2]
# hidden_size = 512


class CaptchaCNN(object):
    def __init__(self, img_width, img_height, num_of_labels, num_of_classes, input_channels,
                 filter_sizes, num_filters, maxpool_sizes, hidden_size):
        """
        :param img_width:
        :param img_height:
        :param num_of_labels:
        :param num_of_classes:
        :param input_channels:
        :param filter_sizes:
        :param num_filters:
        :param maxpool_sizes:
        :param hidden_size:
        """
        self.images = tf.placeholder(tf.float32, shape=[None, img_height, img_width, input_channels], name="input_x")
        self.labels = tf.placeholder(tf.float32, shape=[None, num_of_labels * num_of_classes], name="input_y")
        self._batch_size = tf.shape(self.images)[0]  # int32

        # conv1
        with tf.variable_scope('conv1') as scope:
            filter_shape = (filter_sizes[0], filter_sizes[0], input_channels, num_filters[0])
            kernel = tf.get_variable('weights', filter_shape, tf.float32, initializer=tf.random_normal_initializer())
            kernel = kernel * np.sqrt(2.0 / (filter_shape[0] * filter_shape[1] * filter_shape[3]))
            conv = tf.nn.conv2d(self.images, kernel, [1, 1, 1, 1], padding='SAME')
            # 每种filter有一个统一的bias，因此有多少个filters就设置多少个biases，=out_channels
            biases = tf.get_variable('biases', [filter_shape[3]], tf.float32, initializer=tf.constant_initializer(0.0))
            bias_conv = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias_conv, name=scope.name)

        # pool1
        with tf.variable_scope('maxpool1') as scope:
            pool1 = tf.nn.max_pool(conv1, ksize=[1, maxpool_sizes[0], maxpool_sizes[0], 1],
                                   strides=[1, maxpool_sizes[0], maxpool_sizes[0], 1],
                                   padding='SAME', name=scope.name)

        # conv2
        with tf.variable_scope('conv2') as scope:
            filter_shape = (filter_sizes[1], filter_sizes[1], num_filters[0], num_filters[1])
            kernel = tf.get_variable('weights', filter_shape, tf.float32, initializer=tf.random_normal_initializer())
            kernel = kernel * np.sqrt(2.0 / (filter_shape[0] * filter_shape[1] * filter_shape[3]))
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            # 每种filter有一个统一的bias，因此有多少个filters就设置多少个biases，=out_channels
            biases = tf.get_variable('biases', [filter_shape[3]], tf.float32, initializer=tf.constant_initializer(0.0))
            bias_conv = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias_conv, name=scope.name)

        # pool2
        with tf.variable_scope('maxpool2') as scope:
            pool2 = tf.nn.max_pool(conv2, ksize=[1, maxpool_sizes[1], maxpool_sizes[1], 1],
                                   strides=[1, maxpool_sizes[1], maxpool_sizes[1], 1],
                                   padding='SAME', name=scope.name)

        # conv3
        with tf.variable_scope('conv3') as scope:
            filter_shape = (filter_sizes[2], filter_sizes[2], num_filters[1], num_filters[2])
            kernel = tf.get_variable('weights', filter_shape, tf.float32, initializer=tf.random_normal_initializer())
            kernel = kernel * np.sqrt(2.0 / (filter_shape[0] * filter_shape[1] * filter_shape[3]))
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            # 每种filter有一个统一的bias，因此有多少个filters就设置多少个biases，=out_channels
            biases = tf.get_variable('biases', [filter_shape[3]], tf.float32, initializer=tf.constant_initializer(0.0))
            bias_conv = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(bias_conv, name=scope.name)

        # pool3
        with tf.variable_scope('maxpool3') as scope:
            pool3 = tf.nn.max_pool(conv3, ksize=[1, maxpool_sizes[2], maxpool_sizes[2], 1],
                                   strides=[1, maxpool_sizes[2], maxpool_sizes[2], 1],
                                   padding='SAME', name=scope.name)

        with tf.variable_scope('full-connect-1') as scope:
            num_of_features = int(pool3.get_shape()[1]) * int(pool3.get_shape()[2]) * int(pool3.get_shape()[3])
            flatten_pooled = tf.reshape(pool3, [self._batch_size, num_of_features])
            weights = tf.get_variable('weights', [num_of_features, hidden_size], tf.float32,
                                      initializer=tf.random_normal_initializer())
            weights = weights * np.sqrt(2.0 / (num_of_features * hidden_size))
            biases = tf.get_variable('biases', [hidden_size], tf.float32, initializer=tf.constant_initializer(0.0))
            fc1 = tf.nn.relu(tf.matmul(flatten_pooled, weights) + biases, name=scope.name)

        with tf.variable_scope('output') as scope:
            weights = tf.get_variable('weights', [hidden_size, num_of_chars * num_of_labels], tf.float32,
                                      initializer=tf.random_normal_initializer())
            weights = weights * np.sqrt((hidden_size * num_of_chars * num_of_labels))
            biases = tf.get_variable('biases', [num_of_chars * num_of_labels], tf.float32,
                                     initializer=tf.constant_initializer(0.0))
            outputs = tf.nn.bias_add(tf.matmul(fc1, weights), biases,
                                     name=scope.name)  # shape = [batch_size, num_of_chars*num_of_labels]

        with tf.name_scope('predictions') as scope:
            reshaped_outputs = tf.reshape(outputs, [self._batch_size, num_of_chars, num_of_labels])
            self.predictions = tf.argmax(reshaped_outputs, 2, name=scope)  # shape = [batch_size, num_of_chars]

        with tf.variable_scope('loss') as scope:
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(outputs, self.labels)
            self.loss = tf.reduce_mean(cross_entropy, name=scope.name)

        # Accuracy
        with tf.name_scope("accuracy") as scope:
            reshaped_labels = tf.reshape(self.labels, [self._batch_size, num_of_chars, num_of_labels])
            y_truth = tf.argmax(reshaped_labels, 2, name="y_truth")
            correct_predictions = tf.equal(self.predictions, y_truth)  # [batch_size, num_of_chars]
            self.individual_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
            correct_predictions = tf.reduce_min(tf.cast(correct_predictions, "float"), reduction_indices=1)
            self.accuracy = tf.reduce_mean(correct_predictions, name=scope)

    def index_matrix_to_pairs(self, index_matrix):
        replicated_first_indices = tf.tile(
            tf.expand_dims(tf.range(tf.shape(index_matrix)[0]), dim=1),
            [1, tf.shape(index_matrix)[1]])
        return tf.pack([replicated_first_indices, index_matrix], axis=2)
