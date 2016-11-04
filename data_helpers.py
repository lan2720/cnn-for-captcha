# coding: utf-8

import cv2
import numpy as np
from PIL import Image
from CaptchaGenerator.generate_captcha import init_chars, create_validate_code, binarization

try:
	from cStringIO import StringIO as BytesIO
except ImportError:
	from io import BytesIO


class Dataset(object):
	def __init__(self, extfile=None):
		self.indices = dict(zip(list(init_chars), range(len(init_chars))))
		self.chars = dict(zip(range(len(init_chars)), list(init_chars)))
		self.num_of_labels = len(self.indices)

		if extfile:
			dataset = np.load("150000.npz")
			self.images = dataset['images']
			self.labels = dataset['labels']
			self.count = self.images.shape[0]
		else:
			self.count = 1000000

	# def batch_generator(self, batch_size=50):
	# 	batch_num = self.count / batch_size
	# 	for i in range(batch_num):
	# 		data = []
	# 		labels = []
	# 		for j in range(batch_size):
	# 			code_img, code_str = create_validate_code()
	# 			# code_img.save("{}.jpg".format(code_str))
	# 			binarized_img = binarization(code_img)
	# 			# binarized_img.save("{}_binarized.jpg".format(code_str))
	# 			data.append(self.img2array(binarized_img))
	# 			labels.append(self.get_label(code_str))
	# 		yield np.asarray(data), np.asarray(labels)
	def batch_generator(self, data, labels, batch_size=50, num_of_epoches=100, shuffle=False):
		batch_num = data.shape[0] / batch_size
		for epoch in range(num_of_epoches):
			if shuffle:
				shuffled_indices = np.random.permutation(data.shape[0])
				shuffled_data = data[shuffled_indices]
				shuffled_labels = labels[shuffled_indices]
			else:
				shuffled_data = data
				shuffled_labels = labels
			for i in range(batch_num):
				yield shuffled_data[i * batch_size:(i + 1) * batch_size], shuffled_labels[
																		  i * batch_size:(i + 1) * batch_size]

	def img2array(self, img):
		out = BytesIO()
		img.save(out, format="png")
		out.seek(0)
		img_arr = np.fromstring(out.getvalue(), dtype='uint8')
		img_arr = cv2.imdecode(img_arr, cv2.CV_LOAD_IMAGE_COLOR)
		return img_arr

	def get_label(self, buf):
		a = [self.indices[x] for x in buf]
		one_hot = [0.0] * (len(a) * self.num_of_labels)
		for i in range(len(a)):
			one_hot[i * self.num_of_labels + a[i]] = 1.0
		return np.asarray(one_hot)

	def generate_dataset_on_disk(self):
		for _ in range(150000):
			code_img, code_str = create_validate_code()
			code_img.save("./dataset/{}.jpg".format(code_str))

	def dataset2pkl(self):
		import glob
		images = []
		labels = []
		for fn in glob.glob("./dataset/*.jpg"):
			img_arr = cv2.imread(fn)
			images.append(img_arr)
			label_str = fn.split("/")[-1].split(".")[0]
			label_onehot = self.get_label(label_str)
			labels.append(label_onehot)
		images = np.asarray(images)
		labels = np.asarray(labels)
		np.savez("150000.npz", images=images, labels=labels)

	def test(self):
		im = cv2.imread("4dXQZA.jpg", flags=5)
		print im.shape

		Image.fromarray(im, "RGB").show()


if __name__ == '__main__':
	print("Loading dataset...")
	dataset = Dataset(extfile="150000.npz")
	X = dataset.images  # (149998, 25, 96, 3)
	y = dataset.labels  # (149998, 258)
	X_train, X_test = X[:-100], X[-100:]
	y_train, y_test = y[:-100], y[-100:]
	# # normalize
	# X_mean = np.mean(X_train, axis=0)
	# X_std = np.std(X_train, axis=0)
	# X_train = (X_train - X_mean) / (X_std + 0.00001)
	# X_test = (X_test - X_mean) / (X_std + 0.00001)
	# # generator for trainset and testset
	# train_batch_generator = dataset.batch_generator(X_train, y_train, batch_size=64, shuffle=True)
	# test_batch_generator = dataset.batch_generator(X_test, y_test, batch_size=100, shuffle=False)
	# for x_batch, y_batch in train_batch_generator:
	# 	print x_batch.shape, y_batch.shape
	# 	break
	num = 0
	for x_batch, y_batch in dataset.batch_generator(X_train, y_train, batch_size=64, shuffle=True):
		print num
		num += 1
