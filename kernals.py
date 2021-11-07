import cv2
import numpy as np

identity = np.array([[0,0,0],
                   [0,1,0],
                   [0,0,0]])

box_blur = np.array([[(1./9.),(1./9.),(1./9.)],
                   [(1./9.),(1./9.),(1./9.)],
                   [(1./9.),(1./9.),(1./9.)]])

gauss_3 = np.array([[(1./16.),(2./16.),(1./16.)],
                   [(2./16.),(4./16.),(2./16.)],
                   [(1./16.),(2./16.),(1./16.)]])

gauss_5 = np.array([[(1./16.),(4./16.),(6./16.),(4./16.),(1./16.)],
                    [(4./16.),(16./16.),(24./16.),(16./16.),(4./16.)],
                    [(6./16.),(24./16.),(36./16.),(24./16.),(6./16.)],
                    [(4./16.),(16./16.),(24./16.),(16./16.),(4./16.)],
                    [(1./16.),(4./16.),(6./16.),(4./16.),(1./16.)]])

sharpen = np.array([[0, 1, 0],
                   [1, 5, 1],
                   [0, 1, 0]])

edge_0 = np.array([[1, 0, -1],
                   [0, 0, 0],
                   [-1, 0, 1]])

edge_4 = np.array([[-1, -1, -1],
                   [-1, 4, -1],
                   [-1, -1, -1]])

edge_8 = np.array([[-1,-1,-1],
                   [-1,8,-1],
                   [-1,-1,-1]])

x_edge = np.array([[1,0,-1],
                   [2,0,-2],
                   [1,0,-1]])

y_edge = np.array([[1,2,1],
                   [0,0,0],
                   [-1,-2,-1]])

img = cv2.imread("2.png")

dst = cv2.filter2D(img, -1, identity)
cv2.imshow("0",dst)

dst = cv2.filter2D(img, -1, box_blur)
cv2.imshow("0",dst)

dst = cv2.filter2D(img, -1, gauss_3)
cv2.imshow("0",dst)

dst = cv2.filter2D(img, -1, gauss_5)
cv2.imshow("0",dst)

dst = cv2.filter2D(img, -1, sharpen)
cv2.imshow("0",dst)

dst = cv2.filter2D(img, -1, edge_0)
cv2.imshow("0",dst)

dst = cv2.filter2D(img, -1, edge_4)
cv2.imshow("0",dst)

dst = cv2.filter2D(img, -1, edge_8)
cv2.imshow("0",dst)

img = cv2.imread("circle.png")

dst = cv2.filter2D(img, -1, x_edge)
cv2.imshow("0",dst)

dst = cv2.filter2D(img, -1, y_edge)
cv2.imshow("0",dst)