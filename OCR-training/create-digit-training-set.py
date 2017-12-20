# -*- coding: utf-8 -*-
"""
Create a training set for printed digits.

@author: EM
"""

from PIL import ImageFont, ImageDraw, Image  
import cv2  
import numpy as np  
import matplotlib.pyplot as plt
import os


os.chdir("C:\Users\EM\git-projects\sudoku-solver")

# get all font names
font_folder = "OCR-training\\fonts"
font_paths = os.listdir(font_folder)
font_paths = [os.path.join(font_folder, f) for f in font_paths if f.endswith(".ttf")]


#font_paths = font_paths[:20]


font_nb = len(font_paths)
step = 60 # gap between two digits in px (width and height)



img=Image.new("RGBA", (10*step,font_nb*step),(0,0,0))
draw = ImageDraw.Draw(img)
posy = 0
for font_path in font_paths:
	font = ImageFont.truetype(font_path, 30)
	for i in range(10):
		posx = i * step
		draw.text((posx, posy),str(i),(255,255,255),font=font)
	posy += step

cv2_im = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2GRAY)

plt.imshow(cv2_im[:,:], cmap="gray", interpolation="bicubic")
thresh = cv2.threshold(cv2_im,2,255,cv2.THRESH_BINARY)[1]
plt.imshow(thresh[:1000,:], cmap="gray", interpolation="bicubic")


#i = font_nb-1
#j = 9
#plt.imshow(thresh[step*i:step*(i+1),step*j:step*(j+1)])
#contours = cv2.findContours(thresh[step*i:step*(i+1),step*j:step*(j+1)],
#							 cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

	
def get_bounding_box(img):
	"""
	Get bounding box for contour with max area in img
	Resize result to 10 x 10.
	"""
	contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
	max_area = 0
	max_cnt = None
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if area > max_area:
			max_cnt = cnt
			max_area = area
	if max_cnt is None:
		return(None)
	(bx,by,bw,bh) = cv2.boundingRect(max_cnt)
	roi = img[by:by+bh, bx:bx+bw]
	roi = cv2.resize(roi, (10,10), interpolation=cv2.INTER_AREA)
	return(roi)

roi_list = []
feature_list = []
label_list = []
for i in range(font_nb):
	for j in range(10):
		roi = get_bounding_box(thresh[step*i:step*(i+1),step*j:step*(j+1)])
		if roi is not None:
			feature = roi.reshape((1,100)).astype(np.int16)
			feature_list.append(feature)
			roi_list.append(roi)
			label_list.append(j)

# Check results
test = np.concatenate(roi_list, axis=0)
for i in range(80,90):
	plt.imshow(test[100*i:100*(i+1),:])
	plt.show()
plt.imshow(thresh[15*step:16*step,:])

samples = np.concatenate(feature_list, axis=0)
responses = np.array(label_list)

# Do not keep 0 in training set
idx = (responses != 0)
samples = samples[idx,:]
responses = responses[idx]


# Write training set to file
np.save("OCR-training/generated-samples.npy", samples)
np.save("OCR-training/generated-labels.npy", responses)



