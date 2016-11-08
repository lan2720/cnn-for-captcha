# coding: utf-8

import tensorflow as tf


# filter_sizes = [5, 5, 3]
# num_filters = [32, 32, 32]
# maxpool_sizes = [2, 2, 2]
# hidden_size = 512


class CaptchaCNN(object):
    def __init__(self, filter_sizes, num_of_filters, filter_strides, pool_sizes, pool_strides, pool_types,
                 input_channels, hidden_sizes, num_of_labels, num_of_classes):
        self.filter_sizes = filter_sizes
        self.num_of_filters = num_of_filters
        self.filter_strides = filter_strides
        self.pool_sizes = pool_sizes
        self.pool_strides = pool_strides
        self.pool_types = pool_types
        self.input_channels = input_channels
        self.hidden_sizes = hidden_sizes
        self.num_of_labels = num_of_labels
        self.num_of_classes = num_of_classes

        self.conv_pools = []
        self.full_connects = []
        self._outputs = []

    def conv_pool_layer(self, inputs, filter_size, num_of_filter, filter_stride, maxpool_size, maxpool_stride,
                        pool_type):
        with tf.device('/cpu:0'):
            filter_shape = [filter_size, filter_size, int(inputs.get_shape()[-1]), num_of_filter]
            initializer = tf.contrib.layers.xavier_initializer_conv2d()  # tf.random_normal_initializer(mean=0.0, stddev=0.1)
            filter = tf.get_variable("filter", filter_shape, initializer=initializer)
        # Conv
        conv = tf.nn.conv2d(inputs, filter,
                            strides=[1, filter_stride, filter_stride, 1],
                            padding='VALID',
                            name='conv')
        # Activation
        with tf.device('/cpu:0'):
            bias = tf.get_variable('bias', [num_of_filter], initializer=tf.constant_initializer(0.0))
        bias_conv = tf.nn.bias_add(conv, bias)
        relu_conv = tf.nn.relu(bias_conv, name='relu')
        # Maxpool
        if pool_type == 'max':
            pooled = tf.nn.max_pool(relu_conv, ksize=[1, maxpool_size, maxpool_size, 1],
                                    strides=[1, maxpool_stride, maxpool_stride, 1],
                                    padding='VALID', name='pooling')
        elif pool_type == 'avg':
            pooled = tf.nn.avg_pool(relu_conv, ksize=[1, maxpool_size, maxpool_size, 1],
                                    strides=[1, maxpool_stride, maxpool_stride, 1],
                                    padding='VALID', name='pooling')
        else:
            raise ValueError('pool_type should be `max` or `avg`.')
        return pooled

    def full_connect_layer(self, inputs, weights_shape, biases_shape):
        with tf.device('/cpu:0'):
            weights = tf.get_variable("weights",
                                      weights_shape,
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable("biases",
                                     biases_shape,
                                     initializer=tf.constant_initializer(0.0))
        return tf.matmul(inputs, weights) + biases

    def full_connect_activate(self, inputs, weights_shape, biases_shape, activation='relu'):
        if activation == 'relu':
            return tf.nn.relu(self.full_connect_layer(inputs, weights_shape, biases_shape))
        elif activation == 'sigmoid':
            return tf.nn.sigmoid(self.full_connect_layer(inputs, weights_shape, biases_shape))
        elif activation == 'tanh':
            return tf.nn.tanh(self.full_connect_layer(inputs, weights_shape, biases_shape))
        else:
            raise ValueError('Unsupported activation function, '
                             'please select from `relu`, `sigmoid` and `tanh`.')

    # Main func to build structure
    def flow(self, inputs):
        # stack multi conv layer
        for i in range(len(self.filter_sizes)):
            if i == 0:
                inputs_ = inputs
            else:
                inputs_ = self.conv_pools[-1]
            with tf.variable_scope('conv-pool-{}'.format(i + 1)):
                conv_pool = self.conv_pool_layer(inputs_,
                                                 filter_size=self.filter_sizes[i],
                                                 num_of_filter=self.num_of_filters[i],
                                                 filter_stride=self.filter_strides[i],
                                                 maxpool_size=self.pool_sizes[i],
                                                 maxpool_stride=self.pool_strides[i],
                                                 pool_type=self.pool_types[i])
            self.conv_pools.append(conv_pool)

        # # conv2
        # with tf.variable_scope('conv2') as scope:
        #     filter_shape = (filter_sizes[1], filter_sizes[1], num_filters[0], num_filters[1])
        #     kernel = tf.get_variable('weights', filter_shape, tf.float32, initializer=tf.random_normal_initializer())
        #     kernel = kernel * np.sqrt(2.0 / (filter_shape[0] * filter_shape[1] * filter_shape[3]))
        #     conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        #     # 每种filter有一个统一的bias，因此有多少个filters就设置多少个biases，=out_channels
        #     biases = tf.get_variable('biases', [filter_shape[3]], tf.float32, initializer=tf.constant_initializer(0.0))
        #     bias_conv = tf.nn.bias_add(conv, biases)
        #     conv2 = tf.nn.relu(bias_conv, name=scope.name)
        #
        # # pool2
        # with tf.variable_scope('maxpool2') as scope:
        #     pool2 = tf.nn.max_pool(conv2, ksize=[1, maxpool_sizes[1], maxpool_sizes[1], 1],
        #                            strides=[1, maxpool_sizes[1], maxpool_sizes[1], 1],
        #                            padding='SAME', name=scope.name)

        # stack full connected layers
        for i in range(len(self.hidden_sizes)):
            with tf.variable_scope('full-connect-{}'.format(i + 1)):
                if i == 0:
                    batch_size = int(inputs.get_shape()[0])
                    inputs_ = tf.reshape(self.conv_pools[-1], [batch_size, -1])
                    in_size = int(inputs_.get_shape()[-1])
                else:
                    inputs_ = self.full_connects[-1]
                    in_size = self.hidden_sizes[i - 1]
                full_connect = self.full_connect_activate(inputs_,
                                                          weights_shape=[in_size, self.hidden_sizes[i]],
                                                          biases_shape=[self.hidden_sizes[i]],
                                                          activation='relu')
            self.full_connects.append(full_connect)

        # 由于multilabel任务当前只能使用tf.nn.sigmoid_cross_entropy_with_logits(logits, targets, name=None)计算loss
        # 因此outputs只给出logits，不进行activation
        with tf.variable_scope('outputs') as scope:
            for i in range(self.num_of_labels):
                with tf.variable_scope('part-{}'.format(i)):
                    fc_part = self.full_connect_layer(self.full_connects[-1],
                                                      weights_shape=[self.hidden_sizes[-1], self.num_of_classes],
                                                      biases_shape=[self.num_of_classes])
                    self._outputs.append(fc_part)  # [batch_size, num_of_classes]
            self._outputs = tf.concat(1, self._outputs, name=scope.name)

    @property
    def outputs(self):
        return self._outputs
