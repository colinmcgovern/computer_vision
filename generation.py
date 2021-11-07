import cv2
import os 
import numpy as np
from scipy import ndimage

def arr_to_cv2(arr):
	return np.array(arr*255).astype('uint8')

def m_theta_to_img(G,theta):

	angle = theta
	print(G.max(),G.min())
	print(angle.max(),angle.min())
	print(theta.max(),theta.min())

	M, N = G.shape

	out = np.zeros((M,N,3), np.uint8)

	for i in range(0,M):
		for j in range(0,N):

			#angle 0
			if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
				out[i][j] = (int(G[i][j]),0,0) 
			#angle 45
			elif (22.5 <= angle[i,j] < 67.5):
				out[i][j] = (0,int(G[i][j]),0)
			#angle 90
			elif (67.5 <= angle[i,j] < 112.5):
				out[i][j] = (0,0,int(G[i][j]))
			#angle 135
			elif (112.5 <= angle[i,j] < 157.5):
				out[i][j] = (0,int(G[i][j]),int(G[i][j]))

	return out

def change_brightness(img, value=30):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)
	v = cv2.add(v,value)
	v[v > 255] = 255
	v[v < 0] = 0
	final_hsv = cv2.merge((h, s, v))
	img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
	return img

def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    
    return G,theta

def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D
    angle[angle < 0] += 180
    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z

def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

def hysteresis(img, weak, strong=255):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

def pad(img, kernel):
    r, c = img.shape
    kr, kc = kernel.shape
    padded = np.zeros((r + kr,c + kc), dtype=img.dtype)
    insert = np.uint((kr)/2)
    padded[insert: int(insert + r), insert: int(insert + c)] = img
    return padded

def ApplyMask(image, kernel):
    i, j = kernel.shape
    kernel = np.flipud(np.fliplr(kernel))    
    output = np.zeros_like(image)           
    image_padded = pad(image, kernel)
    for x in range(image.shape[0]):    
        for y in range(image.shape[1]):
            output[x, y] = (kernel * image_padded[x:x+i, y:y+j]).sum()        
    return output

def Gradient_Magnitude(fx, fy):
    mag = np.zeros((fx.shape[0], fx.shape[1]))
    mag = np.sqrt((fx ** 2) + (fy ** 2))
    mag = mag * 255 / mag.max()
    return np.around(mag)

def Gradient_Direction(fx, fy):
    g_dir = np.zeros((fx.shape[0], fx.shape[1]))

    for i in range(0,len(fx)):
    	for j in range(0,len(fx[0])):
    		print(fx[i][j],fy[i][j],np.arctan2(fx[i][j],fy[i][j]))


    g_dir = np.rad2deg(np.arctan2(fy, fx)) + 180
    return g_dir

def calculate_gradient_X(x,y, sigma):
    temp = (x ** 2 + y ** 2) / (2 * sigma ** 2)
    return -((x * np.exp(-temp)) / sigma ** 2)

def calculate_gradient_Y(x,y, sigma):
    temp = (x ** 2 + y ** 2) / (2 * sigma ** 2)
    return -((y * np.exp(-temp)) / sigma ** 2)

def Create_Gx(fx, fy):
    gx = calculate_gradient_X(fx, fy, sigma)
    gx = (gx * 255)
    return np.around(gx)

def Create_Gy(fx, fy):    
    gy = calculate_gradient_Y(fx, fy, sigma)
    gy = (gy * 255)
    return np.around(gy)

def MaskGeneration(T, sigma):
    N = calculate_filter_size(T, sigma)
    shalf = sHalf(T, sigma)
    y, x = np.meshgrid(range(-int(shalf), int(shalf) + 1), range(-int(shalf), int(shalf) + 1))
    return x, y

def calculate_filter_size(T, sigma):
    return 2*sHalf(T, sigma) + 1

def sHalf(T, sigma):
    temp = -np.log(T) * 2 * (sigma ** 2)
    return np.round(np.sqrt(temp))

#Canny example
img = cv2.imread("3.png")
img = cv2.GaussianBlur(img,(3,3), sigmaX=0, sigmaY=0)
img = cv2.Canny(image=img, threshold1=100, threshold2=200) 
cv2.imwrite("canny.png",img)

#DoG example
img = cv2.imread("3.png")
blur1 = cv2.GaussianBlur(img,(3,3), sigmaX=0, sigmaY=0)
blur2 = cv2.GaussianBlur(img,(7,7), sigmaX=0, sigmaY=0)
img = cv2.subtract(blur1,blur2)
cv2.imwrite("dog.png",img)

#Shi Tomasi example
img = cv2.imread("4.png")
corners = cv2.goodFeaturesToTrack(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),200,0.01,10)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(int(x),int(y)),3,(0,0,255),-1)
cv2.imwrite("shi.png",img)

#Canny Edge Detection 

img = cv2.imread("circle.png")
img = cv2.GaussianBlur(img,(3,3), sigmaX=0, sigmaY=0)
cv2.imwrite("step1_canny.png",img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sigma = 0.5
T = 0.3
x, y = MaskGeneration(T, sigma)
gx = -Create_Gx(x, y)
gy = -Create_Gy(x, y)

fx = ApplyMask(img, gx)
fy = ApplyMask(img, gy)
cv2.imwrite("fx.png",arr_to_cv2(fx))
cv2.imwrite("fy.png",arr_to_cv2(fy))
G = Gradient_Magnitude(fx, fy)
theta = Gradient_Direction(fx, fy)
cv2.imwrite("step2_canny_angles.png",m_theta_to_img(G,theta))

non_max_img = non_max_suppression(G,theta)
cv2.imwrite("step3_canny.png",non_max_img)

res,weak,strong = threshold(non_max_img)
cv2.imwrite("step4_canny.png",res)

img = hysteresis(res,weak,strong)
cv2.imwrite("step5_canny.png",img)

print(np.arctan2(0,0))
print(np.arctan2(1,0))
print(np.arctan2(0,1))
print(np.arctan2(1,1))
print(np.arctan2(1,1))
print(np.arctan2(100,100))
