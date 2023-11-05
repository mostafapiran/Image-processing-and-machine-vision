# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 18:16:23 2020

@author: piran_lap
"""

import cv2 
import numpy as np
from scipy import signal as sig
from scipy import ndimage as ndi


img = cv2.imread("pi.jpg",0)
cv2.imshow("img",img)
width , height = img.shape[::-1]

###############################################################################
def gradient_x(imggray):
    ##Sobel operator kernels.
    kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    return sig.convolve2d(imggray, kernel_x, mode='same')
def gradient_y(imggray):
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return sig.convolve2d(imggray, kernel_y, mode='same')

I_x = gradient_x(img)
I_y = gradient_y(img)

Ixx = ndi.gaussian_filter(I_x**2, sigma=3)
Ixy = ndi.gaussian_filter(I_y*I_x, sigma=3)
Iyy = ndi.gaussian_filter(I_y**2, sigma=3)

###############################################################################
k = 0.05
offset=1

# determinant
detA = Ixx * Iyy - Ixy ** 2
# trace
traceA = Ixx + Iyy
    
harris_response = detA - k * traceA ** 2



Sxx=Ixx[offset:height-offset, offset:width-offset]
Syy=Iyy[offset:height-offset, offset:width-offset]
Sxy=Ixy[offset:height-offset, offset:width-offset]

for y in range(offset, height-offset):
    for x in range(offset, width-offset):
        Sxx[y-offset,x-offset] = np.sum(Ixx[y-offset:y+1+offset, x-offset:x+1+offset])
        Syy[y-offset,x-offset] = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])
        Sxy[y-offset,x-offset] = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])

#Find determinant and trace, use to get corner response
det = (Sxx * Syy) - (Sxy**2)
trace = Sxx + Syy
r = det - k*(trace**2)

img_copy_for_corners = np.copy(img)
img_copy_for_edges = np.copy(img)

img_copy_for_corners=cv2.cvtColor(img_copy_for_corners, cv2.COLOR_GRAY2RGB)
img_copy_for_edges=cv2.cvtColor(img_copy_for_edges, cv2.COLOR_GRAY2RGB)

for rowindex, response in enumerate(harris_response):
    for colindex, res in enumerate(response):
        if res > 1000000000:
            # this is a corner
            img_copy_for_corners[rowindex, colindex] =[255,0,0]
        elif res < -10000000:
            # this is an edge
            img_copy_for_edges[rowindex, colindex] =[0,255,0]
            
cv2.imshow("img_corner",img_copy_for_corners)
cv2.imshow("img_edge",img_copy_for_edges)
        






###############################################################################
cv2.waitKey()
cv2.destroyAllWindows()