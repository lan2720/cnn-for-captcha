# coding: utf-8

import tensorflow as tf


# filter_sizes = [5, 5, 3]
# num_filters = [32, 32, 32]
# maxpool_sizes = [2, 2, 2]
# hidden_size = 512


class CaptchaCNN(object):
    def __init__(self, img_width, img_height, num_of_labels, filter_sizes, num_filters, maxpool_sizes, hidden_size=512):
        self.images = tf.placeholder(tf.float32, shape=[None, img_height, img_width, 1], name="input_x")
        self.labels = tf.placeholder(tf.int32, shape=[None, 6], name="input_y")  # 二维list [None, 6]

        with tf.name_scope("conv-1"):
            filter_shape = (filter_sizes[0], filter_sizes[0], 1, num_filters[0])
            filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="filter")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters[0]]), name="b")
            conv = tf.nn.conv2d(self.images, filter, strides=[1, 1, 1, 1], padding='VALID', name='conv')
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled1 = tf.nn.max_pool(h, ksize=[1, maxpool_sizes[0], maxpool_sizes[0], 1], strides=[1, 1, 1, 1],
                                     padding='VALID', name='maxpool')

        with tf.name_scope('conv-2'):
            filter_shape = (filter_sizes[1], filter_sizes[1], num_filters[0], num_filters[1])
            filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="filter")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters[1]]), name="b")
            conv = tf.nn.conv2d(pooled1, filter, strides=[1, 1, 1, 1], padding='VALID', name='conv')
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled2 = tf.nn.avg_pool(h, ksize=[1, maxpool_sizes[1], maxpool_sizes[1], 1], strides=[1, 1, 1, 1],
                                     padding='VALID', name='avgpool')

        with tf.name_scope('conv-3'):
            filter_shape = (filter_sizes[2], filter_sizes[2], num_filters[1], num_filters[2])
            filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="filter")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters[2]]), name="b")
            conv = tf.nn.conv2d(pooled2, filter, strides=[1, 1, 1, 1], padding='VALID', name='conv')
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled3 = tf.nn.avg_pool(h, ksize=[1, maxpool_sizes[2], maxpool_sizes[2], 1], strides=[1, 1, 1, 1],
                                     padding='VALID', name='avgpool')

        # [batch, x, x, num_filters[2]]
        num_of_features = int(pooled3.get_shape()[1]) * int(pooled3.get_shape()[2]) * int(pooled3.get_shape()[3])
        # batch_size = pooled3.get_shape()[0]
        flatten = tf.reshape(pooled3, shape=[-1, num_of_features])

        with tf.name_scope('full-connect-1'):
            # W = tf.Variable(tf.truncated_normal([num_of_features, hidden_size], stddev=0.1), name="W")
            # b = tf.Variable(tf.constant(0.1, shape=[hidden_size]), name="b")
            # fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(W, flatten), b), name="fc1")
            fc1 = self.fullconnected(flatten, num_of_features, hidden_size, name="1")

        with tf.name_scope('full-connect-2'):
            fc21 = self.fullconnected(fc1, hidden_size, num_of_labels, name="21")
            fc22 = self.fullconnected(fc1, hidden_size, num_of_labels, name="22")
            fc23 = self.fullconnected(fc1, hidden_size, num_of_labels, name="23")
            fc24 = self.fullconnected(fc1, hidden_size, num_of_labels, name="24")
            fc25 = self.fullconnected(fc1, hidden_size, num_of_labels, name="25")
            fc26 = self.fullconnected(fc1, hidden_size, num_of_labels, name="26")

        with tf.name_scope('output'):
            self.scores = tf.concat(1, [tf.nn.softmax(fc21), tf.nn.softmax(fc22), tf.nn.softmax(fc23),
                                        tf.nn.softmax(fc24), tf.nn.softmax(fc25), tf.nn.softmax(fc26)])
            # print self.scores.get_shape().as_list()
            self.predictions = tf.pack([tf.argmax(fc21, 1), tf.argmax(fc22, 1), tf.argmax(fc23, 1),
                                        tf.argmax(fc24, 1), tf.argmax(fc25, 1), tf.argmax(fc26, 1)], axis=1)
        # print self.predictions.get_shape().as_list()
        # self.predictions = self.predictions.
        # predictions shape = ? [[9,6,7,4,5,14], [4,8,4,7,2, 19]] 二维

        with tf.name_scope('loss'):
            # self.labels 是每个单独字符的label, 二维[[3,8,5,1,0,2], [4,9,4,1,8,7]] # num_of_labels = 44暂定
            correct_indices = tf.range(0, 6) * num_of_labels + self.labels
            # range得到的是一个一维array，label是一个二维array，broadcast之后correct_indices是一个二维array，
            # self.correct_indices = tf.add(base_indices, self.labels)
            flatten_scores = tf.reshape(self.scores, [1, -1])
            correct_scores = tf.gather(flatten_scores, correct_indices)
            correct_scores = tf.reshape(correct_scores, [-1, 6])
            self.loss = tf.reduce_sum(-tf.log(correct_scores), reduction_indices=1)
            self.reduced_loss = tf.reduce_mean(self.loss)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.cast(self.labels, tf.int64))
            correct_predictions = tf.reduce_min(tf.cast(correct_predictions, "float"), reduction_indices=1)
            self.accuracy = tf.reduce_mean(correct_predictions, name="accuracy")

    def fullconnected(self, data, in_size, num_hidden, name):
        W = tf.Variable(tf.truncated_normal([in_size, num_hidden], stddev=0.1), name="W-" + name)
        b = tf.Variable(tf.constant(0.1, shape=[num_hidden]), name="b-" + name)
        fc = tf.nn.relu(tf.nn.xw_plus_b(data, W, b), name="fc-" + name)
        return fc
