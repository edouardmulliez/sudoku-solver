# -*- coding: utf-8 -*-
"""
Functions to read a sudoku grid and get the digits written on it.

@author: EM
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from Sudoku import solve_sudoku


img_file = "sudoku_capture.JPG" # "sudoku.jpg"


os.chdir("C:\Users\EM\git-projects\sudoku-solver")

##############  Load OCR data for training #######################################
samples = np.float32(np.loadtxt('feature_vector_pixels.data'))
responses = np.float32(np.loadtxt('samples_pixels.data'))

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)


img = cv2.imread(img_file)
if img is None:
	raise ValueError("Could not read image. File exists?")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Filtering image
filtered_img = cv2.GaussianBlur(img, (5,5), 0)

# Converting to black and white
binary_img = cv2.adaptiveThreshold(filtered_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY_INV,11,2)

plt.imshow(binary_img, cmap="gray")

# Getting contours
contours = cv2.findContours(binary_img,
							cv2.RETR_TREE,
							cv2.CHAIN_APPROX_SIMPLE)[1]

# Find the biggest contour (max area)
max_area = 0
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 100:
        peri = cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour, 0.01*peri, True)
        if len(approx) == 4 and area > max_area:
            max_area = area
            max_cnt = contour

img_blank = cv2.multiply(img, np.array([50.0]))
cv2.drawContours(img_blank, [max_cnt], 0, (0,255,0))
#plt.imshow(img_blank, cmap="gray", interpolation="bicubic")


# Resize image to get a perfect square with only the sudoku grid
h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)	# this is corners of new square image taken in CW order

approx = cv2.approxPolyDP(max_cnt,
                          0.01*cv2.arcLength(max_cnt,True),
                          True)[:,0,:]

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

approx = order_points(approx)

M = cv2.getPerspectiveTransform(approx,h)	# apply perspective transformation
warp = cv2.warpPerspective(img, M, (450, 450))


plt.imshow(warp)

############ now take each element for inspection ##############

sudo = np.zeros((9,9),np.uint8)		# a 9x9 matrix to store our sudoku puzzle

smooth = cv2.GaussianBlur(warp,(3,3),5)
thresh = cv2.adaptiveThreshold(smooth,255,0,1,5,2)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
erode = cv2.erode(thresh,kernel,iterations=1)
#erode=thresh
dilate =cv2.dilate(erode,kernel,iterations=2)
contours = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

#plt.imshow(dilate, cmap="gray")



roi_list = []

for cnt in contours:
	area = cv2.contourArea(cnt)
	if True:
#	if 100<area<800:
	
		(bx,by,bw,bh) = cv2.boundingRect(cnt)
#		if (100<bw*bh<1200) and (10<bw<40) and (25<bh<45):
		if (50<bw*bh<2000) and (5<bw<50) and (10<bh<50):
			roi = dilate[by:by+bh,bx:bx+bw]
			small_roi = cv2.resize(roi,(10,10))
			roi_list.append(roi)
			feature = small_roi.reshape((1,100)).astype(np.float32)			
			ret,results,neigh,dist=model.findNearest(feature,k=1)
			integer = int(results.ravel()[0])
			
			gridy,gridx = (bx+bw/2)/50,(by+bh/2)/50	# gridx and gridy are indices of row and column in sudo
			sudo.itemset((gridx,gridy),integer)

len(contours)
len(roi_list)

for roi in roi_list:
	plt.imshow(roi, cmap="gray")
	plt.show()

sum(sum(sudo>0))

# Solve sudoku
solved_grid=solve_sudoku(sudo)

# Print result on image
for i in range(9):
	for j in range(9):
		if sudo[i,j]==0:
			posy,posx = i*50+40,j*50+20
			cv2.putText(warp,str(solved_grid[i,j]),(posx,posy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

cv2.putText(warp,"a",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

plt.imshow(warp, cmap="gray")
plt.show()
