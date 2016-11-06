# coding=utf-8
"""
模仿tensorflow/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
包含：
Dataset类
read_data_sets
"""
from __future__ import print_function
from __future__ import absolute_import

import os
import glob
import cPickle
import shutil
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base
from CaptchaGenerator.generate_captcha import create_validate_code

_letter_cases = "abdefghmnpqrstwxyz"  # 小写字母，去除可能干扰的c i j k l o u v 18
_upper_cases = "ABDEFHMNPQRSTWXYZ"  # 大写字母，去除可能干扰的C G I J K L O U V 17
_numbers = ''.join(map(str, range(2, 10)))  # 数字，去除0，1 (8)
init_chars = ''.join((_letter_cases, _upper_cases, _numbers))
chars = dict(zip(init_chars, range(len(init_chars))))

BATCH_SIZE = 10000
TRAIN_SIZE = 20000
VALIDATION_SIZE = 5000
TEST_SIZE = 10000
DATA_BATCHES_DIR = os.path.join(os.path.dirname(__file__), "data-batches-py")


def dump_batch(filename):
    d = {'data': [], 'labels': []}
    for _ in range(BATCH_SIZE):
        captcha_img, captcha_str = create_validate_code()
        captcha_arr = np.asarray(captcha_img)
        captcha_label = map(lambda i: chars[i], captcha_str)
        d['data'].append(captcha_arr)
        d['labels'].append(captcha_label)
    d['data'] = np.asarray(d['data'])
    d['labels'] = np.asarray(d['labels'])
    cPickle.dump(d, open(os.path.join(DATA_BATCHES_DIR, filename), "wb"))


# 生成数据
def generate_data_sets(train_size=TRAIN_SIZE, test_size=TEST_SIZE):
    if train_size % BATCH_SIZE:
        raise Exception(
            "The value of train_size need to the times of BATCH_SIZE = %d"
            % BATCH_SIZE)
    if test_size % BATCH_SIZE:
        raise Exception(
            "The value of test_size need to the times of BATCH_SIZE = %d"
            % BATCH_SIZE)
    if os.path.exists(DATA_BATCHES_DIR):
        shutil.rmtree(DATA_BATCHES_DIR)
    os.makedirs(DATA_BATCHES_DIR)
    print("Generating raw data, train %d test %d to %s"
          % (train_size, test_size, DATA_BATCHES_DIR))
    # train data
    for i in range(1, (train_size / BATCH_SIZE) + 1):
        filename = "data_batch_{}".format(i)
        dump_batch(filename)
    # test data
    for i in range(1, (test_size / BATCH_SIZE) + 1):
        dump_batch("test_batch")


def unpickle(fn):
    import cPickle
    if isinstance(fn, str):
        fn = open(fn, 'rb')
    assert isinstance(fn, file)
    dict = cPickle.load(fn)
    fn.close()
    return dict


def show_image(img_arr):
    from PIL import Image
    Image.fromarray(img_arr, "RGB").show()


# 主函数
def read_data_sets(data_dir,
                   one_hot=False,
                   validation_size=VALIDATION_SIZE):
    TRAIN_DATA = glob.glob(os.path.join(data_dir, "data_batch_*"))
    TEST_DATA = glob.glob(os.path.join(data_dir, "test_batch"))

    train_images, train_labels = extract_images_and_labels(TRAIN_DATA, one_hot=one_hot)
    test_images, test_labels = extract_images_and_labels(TEST_DATA, one_hot=one_hot)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
                .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images,
                         validation_labels)
    test = DataSet(test_images, test_labels)
    return base.Datasets(train=train, validation=validation, test=test)


def extract_images_and_labels(filenames, one_hot=False):
    images = []
    labels = []
    for fn in filenames:
        with open(fn, 'rb') as f:
            train_batch_images, train_batch_labels = extract_data(f, one_hot=one_hot)
        images.append(train_batch_images)
        labels.append(train_batch_labels)
    images = np.vstack(images)
    labels = np.vstack(labels)
    return images, labels


def dense_to_one_hot(labels_dense, num_classes):
    labels_dim = len(labels_dense.shape)
    if labels_dim == 1:  # 1表示exclusive classification
        num_labels = 1
    elif labels_dim == 2:  # 2表示多label分类问题
        num_labels = labels_dense.shape[1]
    else:
        raise Exception("labels_dense has an invalid dimension %d" % labels_dim)
    num_data = labels_dense.shape[0]
    index_offset = np.arange(num_data * num_labels) * num_classes
    labels_one_hot = np.zeros((num_data, num_labels * num_classes))
    labels_offset = index_offset + labels_dense.ravel()
    labels_one_hot.flat[labels_offset] = 1
    return labels_one_hot


def extract_data(f, one_hot=False):
    print('Extracting', f.name)
    d = unpickle(f)
    if one_hot:
        captcha_onehot = dense_to_one_hot(d['labels'], len(chars))
    else:
        captcha_onehot = d['labels']
    return d['data'], captcha_onehot


class DataSet(object):
    def __init__(self, images, labels):
        """
        Construct a Dataset
        """
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape)
        )
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def main():
    # generate_data_sets()
    datasets = read_data_sets(DATA_BATCHES_DIR, one_hot=True)
    print(datasets.train.labels.shape)
    print(datasets.validation.labels.shape)
    print(datasets.test.labels.shape)


if __name__ == '__main__':
    main()
