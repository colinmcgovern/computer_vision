import cv2
import os 
import numpy as np
from scipy import ndimage

img =  cv2.imread("4.png")

height,width = img.shape

y1 = 0
M = imgheight//12
N = imgwidth//12

cells = []

for y in range(0,height,M):

	row = []

    for x in range(0,width, N):
        y1 = y + M
        x1 = x + N
        cell = im[y:y+M,x:x+N]
        row.append(cell)

    cells.append(row)

