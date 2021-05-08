import os, pygame, cv2, sys
import numpy as np
from itertools import combinations
from scipy.spatial.distance import euclidean
import random


CHAR_IMG = 'target.jpg'
DIR = './pic'
xml_path = '/Users/alan_ford/anaconda3/envs/python37/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml'
file_postfix = ['.jpg', '.jpeg', '.png']
faceCascade = cv2.CascadeClassifier(xml_path)
RANDOM = True
PROB = 30 # percentage of the pixel to be randomized
VALVE = 200


def read_text(text="刘菁我爱你"):
	"""
	Read text -> save character image -> read cv2 image
	"""
	pygame.init()
	font = pygame.font.Font('/System/Library/Fonts/Supplemental/Songti.ttc', 26)
	rtext = font.render(text, True, (0, 0, 0), (255, 255, 255))

	if os.path.exists(CHAR_IMG):
		os.remove(CHAR_IMG)
	pygame.image.save(rtext, CHAR_IMG)
	
	img = cv2.imread(CHAR_IMG)
	img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)/255

	return img


def face_detection(img, faceCascade=faceCascade):
	"""
	load opencv face detection xml from system
	"""
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.2,
		minNeighbors=5,
		minSize=(32, 32))

	# If no face detected
	if len(faces) == 0:
		w = min(img.shape[0], img.shape[1])
		return img[(img.shape[0]-w)//2:(img.shape[0]+w)//2, (img.shape[1]-w)//2:(img.shape[1]+w)//2, :]

	# If faces detected, choose the face with the max size
	max_h, index = 0, 0
	for i, (x, y, w, h) in enumerate(faces):
		if max_h < h:
			max_h, index = h, i

	(x, y, w, h) = faces[index]

	if img.shape[0]>img.shape[1]:
		if x + w/2 < img.shape[0]/2:
			return img[:img.shape[1],:,:]

		else:
			return img[-img.shape[1]:,:,:]

	else:
		if y + h/2 < img.shape[1]/2:
			return img[:,:img.shape[0],:]

		else:
			return img[:,-img.shape[0]:,:]


def read_from_dir(dir, max_size=180):
	"""
	read images from the directory
	"""
	filenames = os.listdir(dir)
	img_list, min_size = [], sys.maxsize
	img_list2 = []

	for filename in filenames:
		# read all the images under directory
		if os.path.join(DIR,filename) == './pic/.DS_Store':
			continue
		img = cv2.imread(os.path.join(DIR,filename))
		img = face_detection(img)
		img_list.append(img)
		min_size = min(img.shape[0], min_size)

	# set the max sub-image size to load into the big picture
	min_size = min(max_size, min_size)
	for img in img_list:
		shrink = cv2.resize(img, (min_size, min_size), interpolation=cv2.INTER_AREA)
		img_list2.append(shrink)
		print(img.shape)
		print(shrink.shape)
		# cv2.imshow('a', shrink)
		# cv2.waitKey(1000)
	return img_list2


def find_max_distance(l):
	"""
	find the max l1-distance in the (1,1,3) array
	"""
	comb = list(combinations(list(range(1, len(l))), 2))
	x, y, max_distance = 0, 0, 0

	for i,j in comb:
		if np.sum(np.abs(l[i]-l[j])) > max_distance:
			x, y, max_distance = i, j, np.sum(np.abs(l[i]-l[j]))
	return x, y, max_distance


def process(dir):
	"""
	img process pip line
	"""
	pixel_list = []
	img_list = read_from_dir(dir = DIR, max_size = 200)
	for img in img_list:
		pixel = np.apply_over_axes(np.mean, img, [0, 1])
		pixel_list.append(pixel)

	pos_1, pos_2, max_dis = find_max_distance(pixel_list)
	vec_a = (pixel_list[pos_2] - pixel_list[pos_1]).reshape(3)
	res = [(0.0, img_list[pos_1]), (1.0, img_list[pos_2])]

	# list the img based on the axis point1 -> point2
	# point1 as 0, point2 as 1, other as the position
	for i in range(len(pixel_list)):
		if i == pos_1 or i == pos_2:
			continue
		pixel = pixel_list[i]
		vec_b = (pixel - pixel_list[pos_1]).reshape(3)
		x = np.matmul(vec_a, vec_b.T)/np.matmul(vec_a, vec_a.T)
		h = vec_b - x * vec_a
		h = np.sum(pow(h,2))
		if h < VALVE:
			res.append((x, img_list[i]))
	res.sort(key=lambda x:x[0])
	return res


def main():
	txt_img = read_text(text='亿周年快乐')
	res = process(dir=DIR)
	factor = res[0][1].shape[0]
	x, y = txt_img.shape[0] * factor, txt_img.shape[1] * factor
	
	new_img = np.zeros((x, y, 3))
	
	for x in range(txt_img.shape[0]):
		for y in range(txt_img.shape[1]):
			# binary search

			pred = 0
			target = txt_img[x, y]
			start, end = 0, len(res)-1
			while start + 1 < end:
				mid = (start + end) // 2
				if res[mid][0] <= target:
					start = mid
				else:
					end = start

			if abs(txt_img[start][0] - target) < abs(txt_img[end][0] - target):
				pred = start
			else:
				pred = end
			
			# randomly choose values around the true value (+1, -1)
			if RANDOM:
				rand = random.randint(1,100)

				if rand < PROB/2 and pred + 1 < len(res) - 1:
					pred += 1
				elif PROB > rand > PROB/2 and pred > 0:
					pred -= 1
			k = res[pred][1][:,:,:]
			new_img[x*factor:(x+1)*factor, y*factor:(y+1)*factor, :] = res[pred][1][:,:,:]
	
	cv2.imwrite('output.png',new_img)


if __name__ == '__main__':
	main()

