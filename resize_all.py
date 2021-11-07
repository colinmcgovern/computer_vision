import cv2
import os
import numpy as np
from scipy import ndimage


#Resizing all images to 256 width max
for img_name in os.listdir("."):
	if(".jpg" in img_name or ".png" in img_name or ".gif" in img_name):

		img = cv2.imread(img_name)

		height, width = img.shape[:2]

		if(width >= 256):
			ratio = float(height)/float(width)

			width = 256
			height = int(ratio * float(width))

			img = cv2.resize(img, (width, height))
			cv2.imwrite(img_name.replace(".jpg",".png"),img)
