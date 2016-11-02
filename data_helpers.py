# coding: utf-8


import numpy as np
from CaptchaGenerator.generate_captcha import init_chars, create_validate_code, binarization


class Dataset(object):
	def __init__(self, data_size):
		self.indices = dict(zip(list(init_chars), range(len(init_chars))))
		self.chars = dict(zip(range(len(init_chars)), list(init_chars)))
		self.num_of_labels = len(self.indices)
		self.count = data_size

	def batch_generator(self, batch_size=50):
		batch_num = self.count / batch_size
		if self.count % batch_num:
			batch_num += 1
		for i in range(batch_num):
			data = []
			labels = []
			for j in range(batch_size):
				code_img, code_str = create_validate_code()
				# binarize and image to array
				binarized_img = binarization(code_img)
				# binarized_img.save("{}_{}.jpg".format(i, j))
				# newimage = code_img.convert('L')
				# newimage.save("{}_{}.jpg".format(i, j))
				data.append(self.img2array(binarized_img))
				labels.append(map(lambda i: self.indices[i], list(code_str)))  # (self.str2onehot(code_str))
			# code str to one-hot array
			yield np.asarray(data), np.asarray(labels)

	def img2array(self, img):
		"""
		:param img: a Image object
		:return: np array
		"""
		width, height = img.size
		return np.array(list(img.getdata())).reshape((height, width, -1)) / 255.0

	def str2onehot(self, label_str):
		"""
		:param str:
		:return: 6 chars and every char has self.num_of_label probability, so the res
				is 6*self.num_of_label one-hot array
		"""
		one_hot = np.zeros((6 * self.num_of_labels,))
		for i in range(len(label_str)):
			idx = self.indices[label_str[i]]
			one_hot[i * self.num_of_labels + idx] = 1
		return one_hot


if __name__ == '__main__':
	dataset = Dataset(data_size=1)
	epochs = 1
	for _ in range(epochs):
		for x, y in dataset.batch_generator(batch_size=1):
			for row in range(25):
				print x[0][row].tolist()
