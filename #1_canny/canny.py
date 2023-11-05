# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:14:05 2020

@author: piran_lap
"""

from scipy import ndimage
import cv2
import numpy as np

img = cv2.imread("helia.jpg",0)
cv2.imshow("img_1",img)

###############################################################################
def gaussian_kernel(size, sigma):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

GU=gaussian_kernel(3,3)
img_c=ndimage.filters.convolve(img, GU)
cv2.imshow("img_c_2",img_c)

###############################################################################
def sobel_filters(img_in):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = ndimage.filters.convolve(img_in, Kx)
    Iy = ndimage.filters.convolve(img_in, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    
    return (G, theta)

img_G,img_T=sobel_filters(img_c)
img_G_8=np.array(img_G, np.uint8)
cv2.imshow("img_G_8_3",img_G_8)

###############################################################################
def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
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

img_non= non_max_suppression(img_G,img_T)
img_non_8=np.array(img_non, np.uint8)
cv2.imshow("img_non_8_4",img_non_8)

###############################################################################
def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.15):
    
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

img_th,th_min,th_max=threshold(img_non)
img_th_8=np.array(img_th, np.uint8)
cv2.imshow("img_th_8_5",img_th_8)

###############################################################################
def hysteresis(img, weak=10, strong=255):
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

img_H=hysteresis(img_th)
img_H_8=np.array(img_H, np.uint8)
cv2.imshow("img_H_8_6",img_H_8)

###############################################################################
cv2.waitKey()
cv2.destroyAllWindows()