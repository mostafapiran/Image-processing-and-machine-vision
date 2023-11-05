# Image processing and machine vision
 My personal projects in machine vision and image processing

# Project - [#1_canny](https://github.com/mostafapiran/Image-processing-and-machine-vision/tree/main/%231_canny)
## Canny Edge Detection in Python
This repository contains Python code for implementing the Canny edge detection algorithm. The Canny edge detector is a popular image processing technique used to detect edges in images. It works by applying a series of image processing steps to identify and highlight the edges in the image while suppressing noise.

## Overview
The code provided in this repository performs the following steps of the Canny edge detection algorithm:

1. **Gaussian Smoothing:** A Gaussian kernel is applied to the input image to reduce noise.
2. **Sobel Filtering:** Sobel filters are used to compute the image gradients and edge magnitudes.
3. **Non-Maximum Suppression:** The algorithm identifies the local maxima in the gradient magnitude to thin out the edges.
4. **Double Thresholding:** Pixels in the gradient magnitude image are categorized as strong, weak, or non-edges.
5. **Edge Tracking by Hysteresis:** Weak edges are connected to strong edges to form continuous edge contours.

## Code Structure
- **gaussian_kernel:** A function to generate a Gaussian kernel for smoothing.
- **sobel_filters:** Calculates gradient magnitudes and orientations using Sobel filters.
- **non_max_suppression:** Implements non-maximum suppression to thin out edges.
- **threshold:** Performs double thresholding to classify edge pixels.
- **hysteresis:** Applies hysteresis to connect weak edges to strong edges.
The code uses the OpenCV and NumPy libraries for image processing.
## Usage
Ensure you have OpenCV and NumPy installed.
Provide your input image by changing the **cv2.imread("helia.jpg", 0)** line.
Run the code to see the Canny edge detection results for your image.


***Feel free to use and modify this code for your edge detection needs. If you find it useful, please consider starring the repository.***
