# coding:utf-8
import os
import math
import random
import shutil
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# abcdefghjkmnpqrstuvwxy
_letter_cases = "abdefghmnpqrstvwxyz"  # 小写字母，去除可能干扰的c i j k l o u v
_upper_cases = "ABDEFHMNPQRSTWXYZ"  # 大写字母，去除可能干扰的C G I J K L O U V
_numbers = ''.join(map(str, range(2, 10)))  # 数字，去除0，1
init_chars = ''.join((_letter_cases, _upper_cases, _numbers))
current_dir = os.path.dirname(__file__)
fontType = os.path.join(current_dir, "luxirb.ttf")
bg_image = os.path.join(current_dir, "background.jpg")
out_dir = os.path.join(current_dir, "mycaptchas")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def create_validate_code(size=(96, 25),
						 chars=init_chars,
						 img_type="jpg",
						 mode="RGB",
						 bg_image=bg_image,
						 fg_color=(255, 255, 255),
						 font_size=19,
						 font_type=fontType,
						 char_length=6,
						 draw_lines=True,
						 n_line=(10, 16),
						 min_length=1,
						 max_length=12,
						 draw_points=False,
						 point_chance=2):
	'''
    @todo: 生成验证码图片
    @param size: 图片的大小，格式（宽，高），默认为(120, 30)
    @param chars: 允许的字符集合，格式字符串
    @param img_type: 图片保存的格式，默认为GIF，可选的为GIF，JPEG，TIFF，PNG
    @param mode: 图片模式，默认为RGB
    @param bg_color: 背景颜色，默认为白色
    @param fg_color: 前景色，验证码字符颜色，默认为白色#FFFFFF
    @param font_size: 验证码字体大小
    @param font_type: 验证码字体，默认为 ae_AlArabiya.ttf
    @param length: 验证码字符个数
    @param draw_lines: 是否划干扰线
    @param n_lines: 干扰线的条数范围，格式元组，默认为(1, 2)，只有draw_lines为True时有效
    @param min_length: 干扰线的最小长度
    @param max_length: 干扰线的最大长度
    @param draw_points: 是否画干扰点
    @param point_chance: 干扰点出现的概率，大小范围[0, 100]
    @return: [0]: PIL Image实例
    @return: [1]: 验证码图片中的字符串
    '''

	width, height = size  # 宽， 高
	img = Image.open(bg_image)  # 创建图形
	draw = ImageDraw.Draw(img)  # 创建画笔
	if draw_lines:
		create_lines(draw, min_length, max_length, n_line, width, height)
	if draw_points:
		create_points(draw, point_chance, width, height)
	strs = create_strs(draw, chars, char_length, font_type, font_size, width, height, fg_color)
	# # 图形扭曲参数
	# params = [1 - float(random.randint(1, 2)) / 100,
	#           0,
	#           0,
	#           0,
	#           1 - float(random.randint(1, 10)) / 100,
	#           float(random.randint(1, 2)) / 500,
	#           0.001,
	#           float(random.randint(1, 2)) / 500
	#           ]
	# img = img.transform(size, Image.PERSPECTIVE, params) # 创建扭曲
	img = img.filter(ImageFilter.DETAIL)  # 滤镜，边界加强（阈值更大）
	return img, strs


def create_lines(draw, min_length, max_length, n_line, width, height):
	'''绘制干扰线'''
	line_num = random.randint(n_line[0], n_line[1])  # 干扰线条数
	for i in range(line_num):
		# 起始点
		begin = (random.randint(0, width), random.randint(0, height))
		# 长度
		length = min_length + random.random() * (max_length - min_length)
		# 角度
		alpha = random.randrange(0, 360)
		# 结束点
		end = (begin[0] + length * math.cos(math.radians(alpha)),
			   begin[1] - length * math.sin(math.radians(alpha)))
		draw.line([begin, end], fill=(255, 255, 255))


def create_points(draw, point_chance, width, height):
	'''绘制干扰点'''
	chance = min(100, max(0, int(point_chance)))  # 大小限制在[0, 100]

	for w in xrange(width):
		for h in xrange(height):
			tmp = random.randint(0, 100)
			if tmp > 100 - chance:
				draw.point((w, h), fill=(0, 0, 0))


def create_strs(draw, chars, char_length, font_type, font_size, width, height, fg_color):
	'''绘制验证码字符'''
	'''生成给定长度的字符串，返回列表格式'''
	# c_chars = random.sample(chars, length) # sample产生的是unique的char
	flag = False
	while not flag:
		c_chars = np.random.choice(list(chars), char_length).tolist()
		strs = ''.join(c_chars)  # 每个字符前后以空格隔开

		font = ImageFont.truetype(font_type, font_size)
		font_width, font_height = font.getsize(strs)

		try:
			start_x = random.randrange(0, width - font_width)
			start_y = random.randrange(0, height - font_height)
		except ValueError as e:
			print e
			print strs
			print width, font_width, height, font_height
		else:
			flag = True

	draw.text((start_x, start_y), strs, font=font, fill=fg_color)
	return ''.join(c_chars)


# def binarization(image):
# 	binarized_img = Image.new("L", size=image.size)
# 	for i in range(image.size[0]):
# 		for j in range(image.size[1]):
# 			r, g, b = image.convert('RGB').getpixel((i, j))
# 			value = int(0.299*r + 0.587*g + 0.114*b)
# 			if value < 180:
# 				binarized_img.putpixel((i, j), 255)
# 			else:
# 				binarized_img.putpixel((i, j), 0)
# 	return binarized_img


if __name__ == "__main__":
	# if os.path.exists(out_dir):
	# 	shutil.rmtree(out_dir)
	# os.makedirs(out_dir)
	# for _ in range(500):
	# 	code_img, code_str = create_validate_code()
	# 	code_img.save("%s/%s.jpg" % (out_dir, code_str))


	# img = Image.open('mycaptchas/2FA8h5.jpg')
	# binarization(img).save("hhh.jpg")

	# binarization_dir = os.path.join(current_dir, "binarization")
	# if os.path.exists(binarization_dir):
	# 	shutil.rmtree(binarization_dir)
	# os.makedirs(binarization_dir)
	# for fn in glob.glob(os.path.join(out_dir, "*.jpg")):
	# 	img = Image.open(fn)
	# 	newfn = fn.split("/")[-1]
	#
	# 	binarization(img).save(os.path.join(binarization_dir, newfn))

	test_image = Image.open(
		"/home/lan/Desktop/test1.jpg")  # /home/lan/PycharmProjects/cnn-for-captcha/CaptchaGenerator/mycaptchas/5WMn6m.jpg
	# binarization(test_image).save("/home/lan/Desktop/test_bi.jpg")
	newimage = test_image.convert('L')  # .save("/home/lan/Desktop/test_bi.jpg")
	print newimage.mode
	print np.array(list(newimage.getdata()))[:100]
