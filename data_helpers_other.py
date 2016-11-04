# coding: utf-8

import cv2
import random
import numpy as np
from PIL import Image
from captcha.image import ImageCaptcha


class Dataset(object):
	def __init__(self, data_size):
		self.captcha = ImageCaptcha(width=96, height=25, font_sizes=[22])
		self.count = data_size
		self.num_of_labels = 10
		self.chars = dict(zip(range(10), map(str, range(10))))

	def gen_rand(self):
		buf = ""
		for i in range(4):
			buf += str(random.randint(0, 9))
		return buf

	def get_label(self, buf):
		a = [int(x) for x in buf]
		one_hot = [0.0] * (len(a) * self.num_of_labels)
		for i in range(len(a)):
			one_hot[i * self.num_of_labels + a[i]] = 1.0
		return np.array(one_hot)

	def gen_sample(self):
		# num = self.gen_rand()
		# img = self.captcha.generate_image(num)
		# img.save("1.jpg")
		# # self.captcha.write(num, 'out.png')
		# # img = np.fromstring(img.getvalue(), dtype='uint8')
		# # img = cv2.imdecode(img, cv2.IMREAD_COLOR)
		# # img = Image()
		# # img = cv2.resize(img, (width, height))
		# # img = np.multiply(img, 1 / 255.0)
		# # img = img.transpose(2, 0, 1)
		# return num, img
		num = self.gen_rand()  # str
		img = self.captcha.generate(num)
		img = np.fromstring(img.getvalue(), dtype='uint8')
		img = cv2.imdecode(img, cv2.IMREAD_COLOR)
		# img = cv2.resize(img, (width, height))
		img = np.multiply(img, 1 / 255.0)
		# img = img.transpose(0, 1, 2)
		return (num, img)

	def batch_generator(self, batch_size=50):
		batch_num = self.count / batch_size
		for _ in range(batch_num):
			data = []
			labels = []
			for _ in range(batch_size):
				label, image = self.gen_sample()
				labels.append(self.get_label(label))  # a list of array
				data.append(image)
			yield np.asarray(data), np.asarray(labels)


def main():
	dataset = Dataset(data_size=10)
	# label, image = dataset.gen_sample()
	# print label
	# print image.shape
	# print image.shape
	for i, j in dataset.batch_generator(batch_size=7):
		print i.shape, j.shape
		print j


if __name__ == '__main__':
	main()
