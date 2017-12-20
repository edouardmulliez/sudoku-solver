# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 22:59:17 2017

@author: EM
"""

# import the necessary packages
import numpy as np
import cv2
 
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

def four_point_transform(image, pts, dimensions=None):
	"""
	Extract a portion of an image between 4 points as a rectangle.
	If dimensions is not None, the dimensions are determined by the maximum 
	height/width of the image defined by the 4 points.
	@param image: np array 
	@param pts: np array of shape (4,2), type int
	@param dimensions: tuple of with 2 int.
	@return: The warped image, with chosen dimensions.
	"""

	rect = order_points(pts)
	
	if dimensions is None:
		(tl, tr, br, bl) = rect
		widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
		widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
		maxWidth = max(int(widthA), int(widthB))
		 
		heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
		heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
		maxHeight = max(int(heightA), int(heightB))
		dimensions = (maxWidth, maxHeight)

	dst = np.array([
		[0, 0],
		[dimensions[0] - 1, 0],
		[dimensions[0] - 1, dimensions[1] - 1],
		[0, dimensions[1] - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, dimensions)
 
	# return the warped image
	return warped