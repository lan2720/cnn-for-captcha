import numpy as np
import tensorflow as tf
from prediction import prediction_for_multi_labels


def test_prediction_for_multi_labels():
    num_of_labels = 4
    num_of_classes = 5
    outputs = np.random.uniform(0.0, 5.0, size=[10, 20])
    outputs = outputs.astype(np.float32)
    predictions = prediction_for_multi_labels(outputs, num_of_labels, num_of_classes)
    print("outputs:", outputs)

    with tf.Session() as sess:
        predictions_ = sess.run(predictions)
        print("predictions:", predictions_)


if __name__ == '__main__':
    test_prediction_for_multi_labels()
