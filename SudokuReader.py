# -*- coding: utf-8 -*-
"""
Functions to read a sudoku grid, solve it and print the missing digits on it.

@author: EM
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

#import os
#os.chdir("C:\Users\EM\git-projects\sudoku-solver")

from perspectiveTransform import four_point_transform
from perfect import solve_sudoku


##############  Load OCR data for training ####################################
samples = np.float32(np.load("OCR-training/generated-samples.npy"))
responses = np.float32(np.load("OCR-training/generated-labels.npy"))

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

###############################################################################


#img_file = "images/sudoku-9.JPG" #     "sudoku.jpg" "sudoku-1.jpg"
#img = cv2.imread(img_file)
#if img is None:
#    raise ValueError("Could not read image. File '"+ img_file +"' exists?")
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#warp = read_solve_display(img)
#plt.imshow(warp)

def read_solve_save(in_file, out_file):
    img = cv2.imread(in_file)
    if img is None:
        raise ValueError("Could not read image. File '"+ in_file +"' exists?")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = crop_black_borders(img)
    status, img = read_solve_display(img)
#    if img is None:
#        print("Could not find grid borders on image.")
#        return;
    cv2.imwrite(out_file, img)
    


def crop_black_borders(img):
    """
    Remove black borders from a picture.
    @param img should be a black and white picture.
    """
    thresh = cv2.threshold(img,1,255,cv2.THRESH_BINARY)[1]
    # Find biggest contour
    contours = cv2.findContours(thresh,
                                cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)[1]
    max_cnt = max([(c, cv2.contourArea(c)) for c in contours], key = lambda t: t[1])[0]
    x,y,w,h = cv2.boundingRect(max_cnt)

    # Crop image
    return(img[y:y+h,x:x+w])

def read_solve_display(img):
    """
    Read an image with a sudoku grid, solves it, display results on image and 
    returns image.
    
    @param img: black and white image as a np array.
    @return: tuple (status, img)
    status gives the results of the reading and solving process. 
    - 0 for read and solved. Then, img is the grid image with the solved sudoku printed on it.
    - 1 for grid borders not found. Then img is an image with the text "Could not find grid borders".
    - 2 for not solved sudoku. That case is probably linked to an error in the
    detection/recognition of the digits. In that case, img is a grid image with
    only the detected digits printed on it.
    """
    
    with_plots=False
    
    # resize image
    img = cv2.resize(img,(350, 350), interpolation = cv2.INTER_AREA)
    
    filtered_img = cv2.GaussianBlur(img, (5,5), 0)
    binary_img = cv2.adaptiveThreshold(filtered_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV,11,2)    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilate = cv2.dilate(binary_img,kernel,iterations = 2)
    
    # Getting contours
    contours = cv2.findContours(dilate,
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
    
    # No grid border found
    if max_area == 0:
        black_image = np.zeros((300,500), dtype=np.int8)
        cv2.putText(black_image,
                    "Could not find grid borders.",
                    (10,170), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255,255,255),2,cv2.LINE_AA)
        plt.imshow(black_image, cmap="gray", interpolation="bicubic")
        return(1,black_image)
    
    if with_plots:
        img_blank = cv2.multiply(img, np.array([50.0]))
        cv2.drawContours(img_blank, [max_cnt], 0, (0,255,0))
        plt.imshow(img_blank, cmap="gray", interpolation="bicubic")
        plt.show()
    
    # Resize image to get a perfect square with only the sudoku grid
    approx = cv2.approxPolyDP(max_cnt,
                              0.01*cv2.arcLength(max_cnt,True),
                              True)[:,0,:]
    warp = four_point_transform(img, approx, dimensions=(450,450))
    if with_plots:
        plt.imshow(warp, cmap="gray", interpolation="bicubic")
        plt.show()
    
    ############ now take each element for inspection ##############
    
    sudo = np.zeros((9,9),np.uint8)        # a 9x9 matrix to store our sudoku puzzle
    
    smooth = cv2.GaussianBlur(warp,(3,3),3)
    thresh = cv2.adaptiveThreshold(smooth,255,0,1,5,2)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    erode = cv2.erode(thresh,kernel,iterations = 1)
    dilate =cv2.dilate(erode,kernel,iterations = 3)
    contours = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    
    if with_plots:
        plt.imshow(dilate, cmap="gray")
        plt.show()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100<area<800:
            (bx,by,bw,bh) = cv2.boundingRect(cnt)
            if (100<bw*bh<1200) and (10<bw<40) and (23<bh<45):
                roi = thresh[by+2:by+bh-2,bx+2:bx+bw-2]
                small_roi = cv2.resize(roi,(10,10))
                feature = small_roi.reshape((1,100)).astype(np.float32)            
                ret,results,neigh,dist=model.findNearest(feature,k=1)
                integer = int(results.ravel()[0])
                gridy,gridx = (bx+bw/2)/50,(by+bh/2)/50    # gridx and gridy are indices of row and column in sudo
                sudo.itemset((gridx,gridy),integer)
    
    sudof= sudo.flatten()
    strsudo = ''.join(str(n) for n in sudof)
    ans = solve_sudoku(strsudo)        # ans is the solved sudoku we get as a string
    
    if not ans:
        # Print read digits on an empty grid
        img = cv2.imread("images/blank_grid.png", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(450,450), interpolation = cv2.INTER_AREA)
        for i in range(9):
            for j in range(9):
                if sudo[i][j] != 0:
                    posy,posx = i*50+35,j*50+20
                    cv2.putText(img,str(sudo[i][j]),
                                (posx,posy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        return(2,img)
    
    # Print result on image
    for i in range(9):
        for j in range(9):
            if sudo[i,j]==0:
                posy,posx = i*50+35,j*50+20
                cv2.putText(warp,str(ans[9*i+j]),(posx,posy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)    
    return(0,warp)


#read_solve_save("images/sudoku-8.jpg", "test.png")
#img = cv2.imread("input.png")
#if img is None:
#    raise ValueError("Could not read image. File '"+ in_file +"' exists?")
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img = crop_black_borders(img)
#status, img = read_solve_display(img)

