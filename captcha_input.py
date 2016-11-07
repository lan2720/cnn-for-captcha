# coding=utf-8
"""
仿照tensorflow/models/image/cifar10/cifar10_input.py的处理方法生产验证码图片数据
将自己的数据集转成Standard TensorFlow format
参考：https://www.tensorflow.org/versions/r0.11/how_tos/reading_data/index.html#file-formats
参考代码：https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/examples/how_tos/reading_data/convert_to_records.py
"""

from __future__ import print_function

import tensorflow as tf
from captcha import read_data_sets
from configs import *


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(value):  # a list of multiple int
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name):
    """将DataSet类中的信息写入TFRecord file"""
    if not os.path.exists(TFRecord_dir):
        os.makedirs(TFRecord_dir)

    images = data_set.images
    labels = data_set.labels
    num_examples = data_set.num_examples

    if images.shape[0] != num_examples:
        raise ValueError('Image size %d does not match dataset size %d.' %
                         (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    # Check some global settings
    if rows != HEIGHT:
        raise ValueError('Image rows %d does not match global param HEIGHT %d.' %
                         (rows, HEIGHT))
    if cols != WIDTH:
        raise ValueError('Image cols %d does not match global param WIDTH %d.' %
                         (cols, WIDTH))
    if depth != NUM_CHANNELS:
        raise ValueError('Image depth %d does not match global param NUM_CHANNELS %d.' %
                         (depth, NUM_CHANNELS))

    filename = os.path.join(TFRecord_dir, name + '.tfrecords')
    print("Writing", filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        label = labels[index].tolist()  # a list of size `num_of_labels` or `num_of_labels*num_of_classes`
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'image_raw': _bytes_feature(image_raw),
            'label': _int64_list_feature(label),
        }))
        writer.write(example.SerializeToString())
    writer.close()


def read_and_decode(filename_queue, one_hot):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Determine the dimension of label vector
    if one_hot:
        label_dim = NUM_OF_LABELS * NUM_OF_CLASSES
    else:
        label_dim = NUM_OF_LABELS

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([label_dim], tf.int64)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)
    return image, label, height, width, depth


# 写数据主函数
def generate_datasets_tfrecords(data_dir, one_hot, validation_size=VALIDATION_SIZE):
    data_sets = read_data_sets(data_dir, one_hot=one_hot, validation_size=validation_size)
    # Convert to Examples and write the result to TFRecords
    convert_to(data_sets.train, 'train')
    convert_to(data_sets.validation, 'validation')
    convert_to(data_sets.test, 'test')


# 读数据主函数
def input_pipeline(data_dir, one_hot, batch_size, num_epochs=None, name=None):
    if not name:
        raise ValueError('Please pass data set name (train/validation/test)')
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(os.path.join(data_dir, name + ".tfrecords")),
        num_epochs=num_epochs,
        shuffle=True
    )
    image, label, height, width, depth = read_and_decode(filename_queue, one_hot=one_hot)
    float_image = tf.cast(image, tf.float32)
    reshaped_image = tf.reshape(float_image, tf.pack([height, width, depth]))
    reshaped_image.set_shape([HEIGHT, WIDTH, NUM_CHANNELS])

    # Preprocessing
    normalized_image = tf.image.per_image_whitening(reshaped_image)

    # min_after_dequeue = 10000
    # capacity = min_after_dequeue + 3 * batch_size
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(TRAIN_SIZE *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    return _generate_images_and_labels_batch(normalized_image, label,
                                             min_queue_examples, batch_size, shuffle=True)


def _generate_images_and_labels_batch(image, label, min_queue_examples,
                                      batch_size, shuffle):
    num_preprocess_threads = 2
    if shuffle:
        image_batch, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples
        )
    else:
        image_batch, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size
        )

    # Display the training images in the visulizer
    tf.image_summary('images', image_batch)
    return image_batch, label_batch


def main():
    # datasets = read_data_sets(DATA_BATCHES_DIR, one_hot=True)
    generate_datasets_tfrecords(DATA_BATCHES_DIR, one_hot=False)

    image_batch, label_batch = input_pipeline(DATA_BATCHES_DIR, one_hot=False,
                                              batch_size=50, num_epochs=10, name="train")
    print(image_batch.get_shape())
    print(label_batch.get_shape())


if __name__ == '__main__':
    main()
