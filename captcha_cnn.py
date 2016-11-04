# coding: utf-8

import tensorflow as tf


# filter_sizes = [5, 5, 3]
# num_filters = [32, 32, 32]
# maxpool_sizes = [2, 2, 2]
# hidden_size = 512


class CaptchaCNN(object):
    def __init__(self, img_width, img_height, num_of_chars, num_of_labels, input_channels, filter_sizes, num_filters,
                 maxpool_sizes, hidden_size):
        """
        :param img_width:
        :param img_height:
        :param num_of_chars: 一张图片中的字符个数,比如4个或6个
        :param num_of_labels: 每种字符可能出现的情况数,比如a-z
        :param filter_sizes:
        :param num_filters:
        :param maxpool_sizes:
        :param hidden_size:
        """
        self.images = tf.placeholder(tf.float32, shape=[None, img_height, img_width, input_channels], name="input_x")
        self.labels = tf.placeholder(tf.int32, shape=[None, num_of_chars], name="input_y")  # 二维list [None, 6]
        self._batch_size = tf.shape(self.images)[0]  # int32

        # conv1
        with tf.variable_scope('conv1') as scope:
            filter_shape = (filter_sizes[0], filter_sizes[0], input_channels, num_filters[0])
            # filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="filter")
            # b = tf.Variable(tf.constant(0.1, shape=[num_filters[0]]), name="b")
            # conv = tf.nn.conv2d(self.images, filter, strides=[1, 1, 1, 1], padding='SAME', name=scope.name)
            # h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # pooled1 = tf.nn.max_pool(h, ksize=[1, maxpool_sizes[0], maxpool_sizes[0], 1], strides=[1, 2, 2, 1],
            # 						 padding='SAME', name='maxpool')

            kernel = self._variable_with_weight_decay('weights',
                                                      shape=filter_shape,
                                                      stddev=5e-2,
                                                      wd=0.0)
            conv = tf.nn.conv2d(self.images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._variable_on_cpu('biases', [num_filters[0]], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope.name)

        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, maxpool_sizes[0], maxpool_sizes[0], 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
        # # norm1
        # norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
        # 				  name='norm1')

        # conv2
        with tf.variable_scope('conv2') as scope:
            filter_shape = (filter_sizes[1], filter_sizes[1], num_filters[0], num_filters[1])
            kernel = self._variable_with_weight_decay('weights',
                                                      shape=filter_shape,
                                                      stddev=5e-2,
                                                      wd=0.0)
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._variable_on_cpu('biases', [num_filters[1]], tf.constant_initializer(0.2))
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope.name)

        # # norm2
        # norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
        # 				  name='norm2')
        # pool2
        pool2 = tf.nn.max_pool(conv2, ksize=[1, maxpool_sizes[1], maxpool_sizes[1], 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # # conv3
        # with tf.variable_scope('conv3') as scope:
        #     filter_shape = (filter_sizes[2], filter_sizes[2], num_filters[1], num_filters[2])
        #     kernel = self._variable_with_weight_decay('weights',
        #                                               shape=filter_shape,
        #                                               stddev=5e-2,
        #                                               wd=0.0)
        #     conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        #     biases = self._variable_on_cpu('biases', [num_filters[2]], tf.constant_initializer(0.1))
        #     bias = tf.nn.bias_add(conv, biases)
        #     conv3 = tf.nn.relu(bias, name=scope.name)
        #
        # # # norm3
        # # norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
        # # 				  name='norm3')
        # # pool3
        # pool3 = tf.nn.max_pool(conv3, ksize=[1, maxpool_sizes[2], maxpool_sizes[2], 1],
        #                        strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        # [batch, x, x, num_filters[2]]
        num_of_features = int(pool2.get_shape()[1]) * int(pool2.get_shape()[2]) * int(pool2.get_shape()[3])
        # batch_size = pooled3.get_shape()[0]
        flatten = tf.reshape(pool2, shape=[-1, num_of_features])

        # with tf.name_scope('full-connect-1'):
        # W = tf.Variable(tf.truncated_normal([num_of_features, hidden_size], stddev=0.1), name="W")
        # b = tf.Variable(tf.constant(0.1, shape=[hidden_size]), name="b")
        # fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(W, flatten), b), name="fc1")
        # local3
        with tf.variable_scope('full-connect-1') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            weights = self._variable_with_weight_decay('weights', shape=[num_of_features, hidden_size],
                                                       stddev=0.04, wd=0.000)
            biases = self._variable_on_cpu('biases', [hidden_size], tf.constant_initializer(0.1))
            fc1 = tf.nn.relu(tf.matmul(flatten, weights) + biases, name=scope.name)
        # fc1 = self.fullconnected(flatten, num_of_features, hidden_size, name="1")

        with tf.variable_scope('full-connect-2') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            weights = self._variable_with_weight_decay('weights', shape=[hidden_size, num_of_chars * num_of_labels],
                                                       stddev=0.04, wd=0.000)
            biases = self._variable_on_cpu('biases', [num_of_chars * num_of_labels], tf.constant_initializer(0.1))
            fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)

        # fc2 = tf.concat(1, [fc21, fc22, fc23, fc24])
        # fc2的每6个neurons进行一次softmax
        self.scores = []
        self.predictions = []
        self.losses = []
        with tf.variable_scope('softmax'):
            # [4, 5, 6, 7]-> 10 labels [0000100000 0000010000 0000001000 0000000100]
            # labels_one_hot = tf.range(0, num_of_chars * self._batch_size) * num_of_labels + flatten_labels
            for i in range(num_of_chars):  # 有多少个字符就需要进行多少次子softmax,比如4 or 6
                sliced_logits = tf.slice(fc2, [0, i * num_of_labels], [self._batch_size, num_of_labels])
                sliced_probs = tf.nn.softmax(sliced_logits)  # [batch, num_of_labels]
                self.scores.append(sliced_probs)  # [batch, 4]
                self.predictions.append(tf.argmax(sliced_probs, 1))  # [batch,]
        self.scores = tf.pack(self.scores, axis=1)  # [batch, num_of_lables*num_of_chars]
        self.predictions = tf.pack(self.predictions, axis=-1)  # [batch, num_of_chars]

        with tf.variable_scope('loss'):
            flatten_scores = tf.reshape(self.scores, [-1, ])  # [batch*num_of_labels*num_of_chars, ]
            flatten_labels = tf.reshape(self.labels, [-1, ])  # labels shape=[batch, num_of_chars]
            correct_indices = tf.range(0, num_of_chars * self._batch_size) * num_of_labels + flatten_labels
            correct_scores = tf.gather(flatten_scores, correct_indices)
            correct_scores = tf.reshape(correct_scores, [self._batch_size, num_of_chars])
            self.loss = tf.reduce_sum(-tf.log(correct_scores), reduction_indices=1)
            self.reduced_loss = tf.reduce_mean(self.loss)

        # with tf.variable_scope('full-connect-21') as scope:
        #     # Move everything into depth so we can perform a single matrix multiply.
        #     weights = self._variable_with_weight_decay('weights', shape=[hidden_size, num_of_labels],
        #                                                stddev=0.04, wd=0.000)
        #     biases = self._variable_on_cpu('biases', [num_of_labels], tf.constant_initializer(0.1))
        #     fc21 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)
        #
        # with tf.variable_scope('full-connect-22') as scope:
        #     # Move everything into depth so we can perform a single matrix multiply.
        #     weights = self._variable_with_weight_decay('weights', shape=[hidden_size, num_of_labels],
        #                                                stddev=0.04, wd=0.000)
        #     biases = self._variable_on_cpu('biases', [num_of_labels], tf.constant_initializer(0.1))
        #     fc22 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)
        #
        # with tf.variable_scope('full-connect-23') as scope:
        #     # Move everything into depth so we can perform a single matrix multiply.
        #     weights = self._variable_with_weight_decay('weights', shape=[hidden_size, num_of_labels],
        #                                                stddev=0.04, wd=0.000)
        #     biases = self._variable_on_cpu('biases', [num_of_labels], tf.constant_initializer(0.1))
        #     fc23 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)
        #
        # with tf.variable_scope('full-connect-24') as scope:
        #     # Move everything into depth so we can perform a single matrix multiply.
        #     weights = self._variable_with_weight_decay('weights', shape=[hidden_size, num_of_labels],
        #                                                stddev=0.04, wd=0.000)
        #     biases = self._variable_on_cpu('biases', [num_of_labels], tf.constant_initializer(0.1))
        #     fc24 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)

        # with tf.variable_scope('full-connect-25') as scope:
        #     # Move everything into depth so we can perform a single matrix multiply.
        #     weights = self._variable_with_weight_decay('weights', shape=[hidden_size, num_of_labels],
        #                                                stddev=0.04, wd=0.000)
        #     biases = self._variable_on_cpu('biases', [num_of_labels], tf.constant_initializer(0.1))
        #     fc25 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)
        #
        # with tf.variable_scope('full-connect-26') as scope:
        #     # Move everything into depth so we can perform a single matrix multiply.
        #     weights = self._variable_with_weight_decay('weights', shape=[hidden_size, num_of_labels],
        #                                                stddev=0.04, wd=0.000)
        #     biases = self._variable_on_cpu('biases', [num_of_labels], tf.constant_initializer(0.1))
        #     fc26 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)

        # fc21 = self.fullconnected(fc1, hidden_size, num_of_labels, name="2-1")
        # fc22 = self.fullconnected(fc1, hidden_size, num_of_labels, name="2-2")
        # fc23 = self.fullconnected(fc1, hidden_size, num_of_labels, name="2-3")
        # fc24 = self.fullconnected(fc1, hidden_size, num_of_labels, name="2-4")
        # fc25 = self.fullconnected(fc1, hidden_size, num_of_labels, name="2-5")
        # fc26 = self.fullconnected(fc1, hidden_size, num_of_labels, name="2-6")


        # with tf.name_scope('softmax'):


        # with tf.name_scope('output'):
        #     # self.scores = tf.concat(1, [tf.nn.softmax(fc21), tf.nn.softmax(fc22), tf.nn.softmax(fc23),
        #     #                             tf.nn.softmax(fc24)]) #, tf.nn.softmax(fc25), tf.nn.softmax(fc26)])
        #     # scores shape = [batch, 44*6]
        #     self.scores = tf.softmax(fc2)#tf.concat(1, [fc21, fc22, fc23, fc24])
        #     self.predictions = tf.pack([tf.argmax(fc21, 1), tf.argmax(fc22, 1), tf.argmax(fc23, 1),
        #                                 tf.argmax(fc24, 1)], axis=1) #, tf.argmax(fc25, 1), tf.argmax(fc26, 1)], axis=1)
        # predictions shape = [batch, 6]
        # print self.predictions.get_shape().as_list()
        # self.predictions = self.predictions.
        # predictions shape = ? [[9,6,7,4,5,14], [4,8,4,7,2, 19]] 二维

        # with tf.name_scope('loss'):
        #     # self.labels 是每个单独字符的label, 二维[[3,8,5,1,0,2], [4,9,4,1,8,7]] # num_of_labels = 44暂定
        #     batch_size = tf.shape(self.images)[0]
        #     flatten_labels = tf.reshape(self.labels, [-1, ])
        #     correct_indices = tf.range(0, 4 * batch_size) * num_of_labels + flatten_labels
        #     flatten_scores = tf.reshape(self.scores, [-1, ])
        #     correct_scores = tf.gather(flatten_scores, correct_indices)
        #     correct_scores = tf.reshape(correct_scores, [-1, 4])
        #     # assert tf.shape(correct_scores)[0] == batch_size
        #     # Use gather_nd, but gradients for tf.gather_nd is not implemented yet.
        #     # correct_indices = tf.range(0, 6) * num_of_labels + self.labels
        #     # correct_indices = self.index_matrix_to_pairs(correct_indices)
        #     # correct_scores = tf.gather_nd(self.scores, correct_indices)
        #     self.loss = tf.reduce_sum(-tf.log(correct_scores), reduction_indices=1)
        #     self.reduced_loss = tf.reduce_mean(self.loss)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.cast(self.labels, tf.int64))
            correct_predictions = tf.reduce_min(tf.cast(correct_predictions, "float"), reduction_indices=1)
            self.accuracy = tf.reduce_mean(correct_predictions, name="accuracy")

    def fullconnected(self, data, in_size, num_hidden, name=None):
        with tf.name_scope("full-connect-{}".format(name)):
            W = tf.Variable(tf.truncated_normal([in_size, num_hidden], stddev=0.1), name="W-" + name)
            b = tf.Variable(tf.constant(0.1, shape=[num_hidden]), name="b-" + name)
            fc = tf.nn.relu(tf.nn.xw_plus_b(data, W, b), name="fc-" + name)
            return fc

    def index_matrix_to_pairs(self, index_matrix):
        replicated_first_indices = tf.tile(
            tf.expand_dims(tf.range(tf.shape(index_matrix)[0]), dim=1),
            [1, tf.shape(index_matrix)[1]])
        return tf.pack([replicated_first_indices, index_matrix], axis=2)

    def _variable_on_cpu(self, name, shape, initializer):
        """Helper to create a Variable stored on CPU memory.
		Args:
		  name: name of the variable
		  shape: list of ints
		  initializer: initializer for Variable
		Returns:
		  Variable Tensor
		"""
        with tf.device('/cpu:0'):
            dtype = tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        """Helper to create an initialized Variable with weight decay.
		Note that the Variable is initialized with a truncated normal distribution.
		A weight decay is added only if one is specified.
		Args:
		  name: name of the variable
		  shape: list of ints
		  stddev: standard deviation of a truncated Gaussian
		  wd: add L2Loss weight decay multiplied by this float. If None, weight
			  decay is not added for this Variable.
		Returns:
		  Variable Tensor
		"""
        dtype = tf.float32
        var = self._variable_on_cpu(
            name,
            shape,
            tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        if wd is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var
