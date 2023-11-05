# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 12:28:00 2020

@author: piran_lap
"""

import numpy as np
import cv2


left= cv2.imread("imL.png",0)
right= cv2.imread("imR.png",0)   
w, h = left.shape[::-1] 

cv2.imshow("img",left)  

depth = np.zeros((w, h), np.uint8)
depth.shape = h, w
    
kernel=6
max_offset=30

kernel_half = int(kernel / 2)    
offset_adjust = 255 / max_offset

for y in range(kernel_half, h - kernel_half):      
    print(".", end="", flush=True)
        
    for x in range(kernel_half, w - kernel_half):
        best_offset = 0
        prev_ssd = 65025
            
        for offset in range(max_offset):               
            ssd = 0
            ssd_temp = 0                            

            for v in range(-kernel_half, kernel_half):
                for u in range(-kernel_half, kernel_half):

                    ssd_temp = int(left[y+v, x+u]) - int(right[y+v, (x+u) - offset])  
                    ssd += ssd_temp * ssd_temp              
                

            if ssd < prev_ssd:
                prev_ssd = ssd
                best_offset = offset
                            
        depth[y, x] = best_offset * offset_adjust

cv2.imwrite('depth1.png',depth)

depth_f = cv2.imread("depth1.png")
cv2.imshow("img_out",depth_f)


cv2.waitKey()
cv2.destroyAllWindows()