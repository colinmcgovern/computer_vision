<div class="main" markdown="1">

#Introduction

Computer vision is the study of how computers can understand digital visual information, like images and videos. Through the use of computer vision we can automate many tasks that the human visual system can do naturally. In this guide, I will use Python when explaining the code parts of computer vision, becuase it is easy to read and most programmers know it. 

Computer vision's understanding of the data can be split into three categories: low, medium, and high. Low level computer vision is at the pixel level and includes image resizing, edge finding, and color segmentation. Medium level computer vision is image level and includes panorama stiching, optical flow, and stereo depth perception. High level computer vision is at the semantic level and includes image tagging, object detection, and instance segmenation. 

#Human Vision

The ability for organisms to percieve light has been around essentially ever since creatures were able to move around. First organisms evolved eyespots, then pit eyes, then finally complex eyes. Acuity. On top of having complex eyes, humans also have the abilty to change their refraction which allows us to focus. Rods and cones. How we percieve depth. Fovea. Eyes need movement to see. Fixational eye movement. Human colorspace. Our trichromancy and what colors we percieve stronger. Some things easy for human vision is difficult for computer vision and vise versa.

#Image Basics

-what is an image
-lens basics: lens, focal length, ...
-cartesian vs computer screen coords 
-color images to a computer algorithm is a 3d tensor
-row major vs column major
-HWC vs CHW
-Triangle, bilinear, and bicubic interpoliation

An image is when a 3D space is projected onto a 2D plane.

![Pinhole Camera](pinhole.png)

#Resizing 

-how to make smaller
-how to make larger
-artifacts with bilinear and bicubic interpolation

#Kernels

Kernels in computer vision is a small matrix convolved over an image to do a number of useful changes, for example, blurring, sharpening, and edge detection. The convolution with the kernal over an image is done by mulitplying its weights on each possible pixel of the image with a stride of 1. Unless you add in padding before the kernal convolution, it will result in N-1 smaller image. For example, a 3*3 kernal passed over a 16*16 image would result in a 14*14 sized image. Kernels which have even sized dimentions, for example 2*2 or 4*4, give bad results and therefore are never used.

In the example below, you can see how an edge detection kernel convoled over a 16*16 image is calculated. The edge detection filter we are going to use is:
![4_edge_kernel](4_edge_kernel.png)
<br>

|Step #|Multiplication On Image            |Resulting Pixel                        |Output Image                       |
|------|-----------------------------------|---------------------------------------|-----------------------------------|
|1     |![kernel_mul_1](kernel_mul_1.png)  |![kernel_pixel_1](kernel_pixel_1.png)  |![kernel_out_1](kernel_out_1.png)  |
|2     |![kernel_mul_2](kernel_mul_2.png)  |![kernel_pixel_2](kernel_pixel_2.png)  |![kernel_out_2](kernel_out_2.png)  |
|...   |...                                |...                                    |...                                |
|14    |![kernel_mul_14](kernel_mul_14.png)|![kernel_pixel_14](kernel_pixel_14.png)|![kernel_out_14](kernel_out_14.png)|

**Types of Kernels**

|Name                     |Kernel|Result|Description                            |
|-------------------------|------|------|---------------------------------------|
|Identity                 |![identity_kernel](identity_kernel.png)|![identity_kernel_out](identity_kernel_out.png)|Does nothing except for shrink the resulting image's height and width down by 2 pixels.|
|Box Blur                 |![box_blur_kernel](box_blur_kernel.png)|![box_blur_kernel_out](box_blur_kernel_out.png)|Filters out high frequency details out an image, but this usually in practice Gaussian blurs are used.  |
|3*3 Guassian Blur        |![3_gauss_kernel](3_gauss_kernel.png)|![3_gauss_kernel_out](3_gauss_kernel_out.png)|Filters out high frequency details out of an image.|
|5*5 Guassian Blur        |![5_gauss_kernel](5_gauss_kernel.png)|![5_gauss_kernel_out](5_gauss_kernel_out.png)|Filters out more high frequency details out of an image.|
|Sharpening               |![sharpen_kernel](sharpen_kernel.png)|![sharpen_kernel_out](sharpen_kernel_out.png)|Increases the sharpness of the pixels in the image.|
|0 Centered Edge Detection|![0_edge_kernel](0_edge_kernel.png)|![0_edge_kernel_out](0_edge_kernel_out.png)|Filters out low frequency details out of an image|
|4 Centered Edge Detection|![4_edge_kernel](4_edge_kernel.png)|![4_edge_kernel_out](4_edge_kernel_out.png)|Filters out more low frequency details out of an image|
|8 Centered Edge Detection|![8_edge_kernel](8_edge_kernel.png)|![8_edge_kernel_out](8_edge_kernel_out.png)|Filters out even more low frequency details out of an image|

##LoG and DoG Filters

**WRITE THIS NOT FOR AWHILE. I NEED TO DECIDE ON BLOB DETECTION FIRST**

|Filter                 |Image                |
|-----------------------|---------------------|
|Original               |![2](2.png)          |
|Simple Edge Detection  |![2_edge](2_edge.png)|
|Lapacian of Gausians   |![2_log](2_log.png)  |
|Difference of Gaussians|![2_dog](2_dog.png)  |

##Sobel Edge Detection

Sobel edge detection is an algorithm which is used in computer vision to find not only the edges in an image, but also the edges' magnitude in direction. We do this by using the vertical and horizontal edge kernels. By finding the hypotenuse of the vertical and horizontal edges, we get the edges' magnitude. By plugging the vertical and hortizontal  edges into atan2(), we get the edges' direction. The edge detection is between 0 and 180&deg;.

As an example, let us apply Sobel Edge Detection to the following image:
![circle](circle.png)

|Input(s)                                      |Operation                |Output                             |
|----------------------------------------------|-------------------------|-----------------------------------|
|![circle](circle.png)                         |![x_kernel](x_kernel.png)|![circle_x](circle_X.png)          |
|                                              |                         |Formally we call this G<sub>x</sub>|
|![circle](circle.png)                         |![y_kernel](y_kernel.png)|![circle_y](circle_y.png)          |
|                                              |                         |Formally we call this G<sub>y</sub>|
|![circle_x](circle.png)![circle](circle_y.png)|![hyp](hyp.png)          |![circle_hyp](circle_hyp.png)      |
|![circle_x](circle.png)![circle](circle_y.png)|![atan2](atan2.png)      |![circle_atan2](circle_atan2.png)  |

By visualizing the x angle, 90&deg;, with a red hue and the y angle, 0&deg;, with a blue hue, and the magnitude of the edge with the value, we get the following image:
![circle_sobel](circle_sobel.png)

#Features

In computer vision, a feature is any piece of information on an image that is relevent to the problem we are trying to solve. Common features to look for are edges, blobs, and corners.

##Edge Detection

Edge detection is any mathematical method that aims at identfying edges or curves in an image at which the image brightness changes sharply. The most common algorithm used for modern edge detection is the "Canny edge detector". It works in the following five steps:

1. Reduce noise using a gaussian filter

    Edge detection is very sensitive to noise so we apply a gaussian filter with the size we feel works best. In the example below I applied a 5 * 5 filter.

    ![step1_canny](step1_canny.png)

2. Find the intensity gradients of the image

    Second we want to find the intesity gradients of the image. We do this by applying the vertical edge detector and the horzonatal edge detector to make two separate matrices. Using these two matricies we find the magnitudes of the edges via the hypotenuse function and the angles of the edges via the arctan() function.

    ![step2_canny_angles](step2_canny_angles.png)
*Angles*

    ![step2_canny_m](step2_canny_m.png)
*Magnitudes*

3. Apply non-max supression to thin the egdes

    Next, we want to thin the edges. We do this by comparing each pixel's magnitude to the magnitude of it's neighbor's pixels in the same direction as it's angle. If the pixel has the largest magnitude compared to its neighboors, it remains the same, if not we set its magnitude to zero. For example, if a pixel has a vertical angle we compare it's magnitude to the pixel above and below it.

    ![step3_canny](step3_canny.png)

4. Double threshold to filter pixels into three categories

    After thinning the edges, we want to split the pixels into three categories: strong, weak, and very weak. To do this we set two thresholds: one as the minimum threshold to be considered a "strong" pixel and another to be considered a "weak" pixel. Any pixel who's gradient value is greater than the first threshold is kept in the final output. Any pixel between the two thresholds is a weak pixel and may or may not be in the final output. Any pixel below both thresholds will not be in the final output.

    ![step4_canny](step4_canny.png)
    *Strong pixels in red, weak pixels in green, and very weak pixels in black*

5. Perform hysterious to filter pixels into two categories

    The last step is deciding which of the weak pixels from the last step to keep. If a "weak" pixel has a strong neighbor as one of its 8 immediate neighbors (North, South, East, West, North-East, etc), it is kept in the final output and if it doesn't it is considered noise and it is removed.

    ![step5_canny](step5_canny.png)

##Blob Detection

Blob detection in computer vision are methods used to detect contiguous regions of an image that are differ in properties, like color or brightness. When using a high pass filter to find places of difference in images, sometimes this can result in the high frequency noise being amplified. For this reason we have the LoG (Lapacian of Guassians) and DoG (Difference of Gaussians) filters. Both of these filters are shaped similar to Mexican hats and are used for finding regions of difference in an image without amplying the noise. **THIS NEEDS TO BE REWRITTEN AFTER NEARLY EVERYTHING ELSE IS DONE**.

##Corner Detection
Corners in images are very nice to know in computer vision because they have high self-difference and therefore are likely unique to that section of the image. Which directionever we move the image patch it is different to its proximal patches. If we keep track of only the corner patches, we can significantly improve algorithms for video tracking, panorama stitching, etc. A 1920 by 1080 image has ~2 million 11 by 11 patches, but a corner detection algorithm could cut that down to 100 or however much you need patches. 

Lets use the following image of a square to demonstrate why use corners for features we track:

![square](square.png) 

|Type of Patch  |Picture                                |Moved Right                              |Moved Up                               |Self-Difference|
|---------------|---------------------------------------|-----------------------------------------|---------------------------------------|---------------|
|No Edges       |![square_no_edges](square_no_edges.png)|![square_no_edges](square_no_edges.png)  |![square_no_edges](square_no_edges.png)|Terrible       |
|Horizontal Edge|![square_hor](square_hor.png)          |![square_hor_right](square_hor_right.png)|![square_hor_up](square_hor_up.png)    |Okay           |
|Vertical Edge  |![square_vert](square_vert.png)        |![square_vert](square_vert_right.png)    |![square_vert](square_vert_up.png)     |Okay           |
|Corner         |![square_corner](square_corner.png)    |![square_corner](square_corner_right.png)|![square_corner](square_corner_up.png) |Great          |

The best algorithm for this introduced in 1988 was the Harris corner detector, then in 1994 a slight modification to this algorithm that dramatically improved results is the Shi-Tomasi algorithm. The Shi-Tomasi algorithm is so good its function in the CV2 library is called "goodFeaturesToTrack()".

|                       |Image                |
|-----------------------|---------------------|
|Original               |![4](4.png)          |
|Harris-Corner Detection|![harris](harris.png)|
|Shi-Tomasi             |![shi](shi.png)      |

##Feature Descriptors

A feature descriptor is any algorithm which takes in an image and outputs a list of features that encode interesting information about an image.

###SIFT (Scale Invariant Feature Transform)

SIFT is a patented 

###HOG (Histogram of Gradients)

The HOG algorithm splits an image into a grid and then finds the average gradient of each cell. HOG combined with a SVM (Support Vector Machine) before neural networks was one of the best methods of object detection. How it works is that after splitting the image into cells, a histrogram of the direction of each cell's gradients are calculated and the most common gradient in that cell is chosen. In order to account for the changes in illumination and contrast, we can perform "block normalization" where we have each histogram be relative to its neighboring blocks. 

|                      |Image                  |
|----------------------|-----------------------|
|Original image        |![4](4.png)            |
|Image split into cells|![4_cells](4_cells.png)|

|      |Image                    |Magnitude and Gradients                    |Histogram                          |Output                           |
|------|-------------------------|-------------------------------------------|-----------------------------------|---------------------------------|
|Cell 1|![4_cell_1](4_cell_1.png)|![4_mag_grad_cell_1](4_mag_grad_cell_1.png)|![4_hist_cell_1](4_hist_cell_1.png)|![4_out_cell_1](4_out_cell_1.png)|
|Cell 2|![4_cell_2](4_cell_2.png)|![4_mag_grad_cell_2](4_mag_grad_cell_2.png)|![4_hist_cell_2](4_hist_cell_2.png)|![4_out_cell_2](4_out_cell_2.png)|
|...   |...                      |...                                        |...                                |...                              |

The following is the HOG features on an image before normalization:
![4_hog](4_hog.png)

|      |Image                    |Histogram                          |Neighbors                          |Neighbor Histogram                           |Histogram Normalized                         |Normalized Output                          |
|------|-------------------------|-----------------------------------|-----------------------------------|---------------------------------------------|---------------------------------------------|-------------------------------------------|
|Cell 1|![4_cell_1](4_cell_1.png)|![4_hist_cell_1](4_hist_cell_1.png)|![4_cell_1_neighbors](4_cell_1.png)|![4_cell_1_neighbors_hist](4_cell_1_hist.png)|![4_hist_cell_1_norm](4_hist_cell_1_norm.png)|![4_out_cell_1_norm](4_out_cell_1_norm.png)|
|Cell 2|![4_cell_1](4_cell_1.png)|![4_hist_cell_2](4_hist_cell_2.png)|![4_cell_2_neighbors](4_cell_2.png)|![4_cell_2_neighbors_hist](4_cell_2_hist.png)|![4_hist_cell_2_norm](4_hist_cell_2_norm.png)|![4_out_cell_2_norm](4_out_cell_2_norm.png)|
|...   |...                      |...                                |...                                |...                                          |...                                          |...                                        |

The following is the final output of the HOG algorithm:
![4_hog_normal](4_hog_normal.png)

###ORB

#Example Problem: Stitching Together a panorama

-RANSAC
-homography 
-cylinder projection

#Optical Flow

-sparse vs dense
-problems with feature tracking
-lucas kanade
-aperature problem
-image pyramids
-farneback 

#Depth 

-we percive depth: binocoluar covergence, bi parallax, monocular movement parallax, image cues
-computers percieve depth: stereo cameras, lidar, structured light, time of flight
-depth is used for object seg, 3d reconstruction, facial regoniztion, pose tracking

##Stereo depth
-persepective projection 
-camera intrinsics
-how to solve for depth with stereo camera
-graph cut
-assumptions which are pitfalls: uniqueness, ordering, and smoothness

<div class="footer">2021 Colin McGovern. Feel free to copy.</div>
</div>