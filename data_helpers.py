# coding: utf-8


import numpy as np
from CaptchaGenerator.generate_captcha import init_chars, create_validate_code


class Dataset(object):
	def __init__(self):
		self.chars = dict(zip(list(init_chars), range(len(init_chars))))
		self.num_of_labels = len(self.chars)

	def batch_generator(self, batch_size=50, batch_num=20000):
		for _ in range(batch_num):
			data = []
			labels = []
			for _ in range(batch_size):
				code_img, code_str = create_validate_code()
				# print "1st line:", [code_img.convert('RGB').getpixel((i, 0)) for i in range(96)]
				# image to array
				data.append(self.img2array(code_img))
				labels.append(map(lambda i: self.chars[i], list(code_str)))  # (self.str2onehot(code_str))
			# code str to one-hot array
			yield np.asarray(data), np.asarray(labels)

	def img2array(self, img):
		"""
		:param img: a Image object
		:return: np array
		"""
		width, height = img.size
		newimage = img.convert('L')
		return np.array(list(newimage.getdata())).reshape((height, width, -1)) / 255.0

	def str2onehot(self, label_str):
		"""
		:param str:
		:return: 6 chars and every char has self.num_of_label probability, so the res
				is 6*self.num_of_label one-hot array
		"""
		one_hot = np.zeros((6 * self.num_of_labels,))
		for i in range(len(label_str)):
			idx = self.chars[label_str[i]]
			one_hot[i * self.num_of_labels + idx] = 1
		return one_hot


if __name__ == '__main__':
	dataset = Dataset()
	epochs = 1
	for _ in range(epochs):
		for x, y in dataset.batch_generator(batch_size=1, batch_num=1):
			print x.shape, y.shape
