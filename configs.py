# coding=utf-8

import os

# Captcha
_letter_cases = 'abdefghmnpqrstwxyz'  # 小写字母，去除可能干扰的c i j k l o u v 18
_upper_cases = 'ABDEFHMNPQRSTWXYZ'  # 大写字母，去除可能干扰的C G I J K L O U V 17
_numbers = ''.join(map(str, range(2, 10)))  # 数字，去除0，1 (8)
init_chars = ''.join((_letter_cases, _upper_cases, _numbers))
chars = dict(zip(init_chars, range(len(init_chars))))

NUM_OF_LABELS = 6
NUM_OF_CLASSES = len(chars)

# Picture
HEIGHT = 25
WIDTH = 96
NUM_CHANNELS = 3

# Data
BATCH_SIZE = 10000
TRAIN_SIZE = 150000
VALIDATION_SIZE = 0
TEST_SIZE = 10000
DATA_BATCHES_DIR = os.path.join(os.path.dirname(__file__), "data-batches-py")
TFRecord_dir = os.path.join(os.path.dirname(__file__), "tfrecords")
