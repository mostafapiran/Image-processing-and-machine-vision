# Image processing and machine vision
 My projects in machine vision and image processing

# Project - [#1_canny](https://github.com/mostafapiran/Image-processing-and-machine-vision/tree/main/%231_canny)
---

<p align="center">
 <img src="https://github.com/mostafapiran/Image-processing-and-machine-vision/blob/main/%231_canny/2.png">

</p>

# Canny Edge Detection in Python using OpenCV and SciPy

## Introduction

This repository contains a Python script for performing Canny Edge Detection, a popular technique in computer vision for detecting edges in images. The Canny edge detector is widely used in applications such as image segmentation, object recognition, and feature extraction.

## Code Overview

The Python script provided in this repository performs the following steps:

1. **Image Preprocessing:** It reads an input image (`helia.jpg`) and converts it to grayscale, preparing it for edge detection.

2. **Gaussian Smoothing:** The script applies Gaussian smoothing to the image to reduce noise and make the edges more distinct.

3. **Gradient Calculation:** It computes the image gradients using Sobel operator kernels to capture changes in intensity.

4. **Non-Maximum Suppression:** The code identifies the local maxima in the gradient magnitude, preserving only the strongest edges.

5. **Double Thresholding:** A double thresholding technique is used to categorize edge pixels as strong, weak, or non-edges based on their gradient magnitude values.

6. **Edge Tracking by Hysteresis:** Finally, the script applies hysteresis to connect weak edges to strong edges, ensuring a continuous representation of the detected edges.

7. **Visualization:** Detected edges are visualized and displayed using OpenCV.

## Usage

To use this code for Canny Edge Detection:

1. Ensure you have OpenCV, NumPy, and SciPy installed in your Python environment.

2. Replace the path to the input image by modifying the line: `img = cv2.imread("helia.jpg", 0)`

3. Adjust the parameters such as `lowThresholdRatio` and `highThresholdRatio` to control the edge detection thresholds.

4. Run the script to visualize the detected edges in the input image.

## Contribution and Modification

You are welcome to use, modify, and contribute to this code for your edge detection needs. If you find it valuable, consider giving this repository a star.

Feel free to explore the code, experiment with different images, and adapt it to your specific use cases. We hope this implementation helps you better understand and apply Canny Edge Detection in your projects.

## Acknowledgment

We acknowledge the open-source community and the contributors to OpenCV and SciPy, which make it possible to share and collaborate on code like this.

---
# Project - [#2_corner](https://github.com/mostafapiran/Image-processing-and-machine-vision/tree/main/%232_corner)
---

# Harris Corner Detection in Python using OpenCV and SciPy

## Introduction

This repository contains a Python script for performing Harris Corner Detection, a classic computer vision technique for identifying corner points or interest points in images. Corners are essential features used in various computer vision tasks, including image stitching, object tracking, and feature matching.

## Code Overview

The Python script provided in this repository carries out the following steps:

1. **Image Preprocessing:** It reads an input image (`pi.jpg`) and converts it to grayscale, simplifying the processing of corner detection.

2. **Gradient Computation:** The script computes image gradients using Sobel operator kernels to capture variations in intensity.

3. **Structure Tensor Calculation:** It calculates the elements of the structure tensor by applying Gaussian smoothing to the gradient products. This step is crucial for understanding the local image structure.

4. **Corner Response:** The Harris corner response is computed for each pixel in the image. This response combines information about local gradients and their variations, helping identify corners.

5. **Corner Classification:** The code classifies pixels based on their Harris corner response. Pixels with high positive responses are identified as corners, while negative responses indicate edges.

6. **Visualization:** Detected corners and edges are visually highlighted in the input image, providing a clear and intuitive representation of the results.

## Usage

To use this code for Harris Corner Detection:

1. Ensure you have OpenCV, NumPy, and SciPy installed in your Python environment.

2. Replace the path to the input image by modifying the line: `img = cv2.imread("pi.jpg", 0)`

3. Adjust the `k` and `offset` parameters to fine-tune the detection criteria based on your specific image and requirements.

4. Run the script to visualize the detected corners and edges in the input image.

## Contribution and Modification

You are welcome to use, modify, and contribute to this code for your corner detection needs. If you find it valuable, consider giving this repository a star.

Feel free to explore the code, experiment with different images, and adapt it to your specific use cases. We hope this implementation helps you better understand and apply Harris Corner Detection in your projects.

## Acknowledgment

We acknowledge the open-source community and the contributors to OpenCV and SciPy, which make it possible to share and collaborate on code like this.

---
# Project - [#3_ferquency matching-matchTemplate](https://github.com/mostafapiran/Image-processing-and-machine-vision/tree/main/%233_ferquency%20matching-matchTemplate)
---

<p align="center">
 <img src="https://github.com/mostafapiran/Image-processing-and-machine-vision/blob/main/%233_ferquency%20matching-matchTemplate/4.png">
</p>

# Template Matching in Python using OpenCV

## Introduction

This repository contains a Python script for performing template matching using the OpenCV library. Template matching is a common computer vision technique used to locate a template (sub-image) within a larger image. It's widely used for tasks like object detection and image recognition.

## Code Overview

The Python script provided in this repository carries out the following steps:

1. **Loading Images:** It reads the input image and the template image. The input image can be any image, and the template is the smaller image you want to find within the larger one.

2. **Image Conversion:** The input image is converted to grayscale to simplify the matching process.

3. **Template Matching:** Template matching is performed using the `cv2.matchTemplate` function with the `cv2.TM_CCOEFF_NORMED` method. This method identifies regions in the input image that closely match the template.

4. **Thresholding:** The script sets a similarity threshold (0.9) to identify the best matches. Regions with a similarity score above this threshold are considered matches.

5. **Visualization:** The script highlights the identified matches in the input image by drawing rectangles around them.

6. **Saving Output:** The script saves the output image with the detected matches as 'res.png'.

## Usage

To use this code for template matching:

1. Ensure you have OpenCV installed in your Python environment.

2. Replace the paths to the input image and the template image by modifying the lines:
   - `img_rgb = cv2.imread('full s.jpg')`
   - `template = cv2.imread('i.jpg',0)`

3. Adjust the similarity threshold (`threshold`) to fine-tune the matching criteria.

4. Run the script to perform template matching and generate the output image with highlighted matches.

## Contribution and Modification

You are welcome to use, modify, and contribute to this code for your template matching needs. If you find it valuable, consider giving this repository a star.

Feel free to explore the code, experiment with different images, and adapt it to your specific use cases. We hope this implementation helps you with your template matching projects.

## Acknowledgment

We acknowledge the open-source community and the contributors to OpenCV, which makes it possible to share and collaborate on code like this.

---
# Project - [#4_stereo matching](https://github.com/mostafapiran/Image-processing-and-machine-vision/tree/main/%234_stereo%20matching)
---

# Stereo Depth Map Generation using Block Matching

## Introduction

This repository contains a Python script for generating a stereo depth map from a pair of stereo images using block matching. Stereo depth maps provide information about the 3D structure of a scene and are commonly used in computer vision applications, such as stereo vision, object recognition, and autonomous navigation.

## Code Overview

The Python script provided in this repository performs the following steps:

1. **Loading Stereo Images:** It reads two stereo images (`imL.png` and `imR.png`) that form a stereo pair. These images represent the same scene from different viewpoints.

2. **Image Preprocessing:** The images are converted to grayscale to simplify the depth map generation process.

3. **Block Matching:** The core of the algorithm is block matching. It slides a window of a defined size (`kernel`) across the images and finds the best matching block in the right image for each block in the left image. The offset (disparity) is computed for each block, representing the difference in horizontal position between the corresponding blocks in the two images.

4. **Depth Map Creation:** The script generates a depth map based on the computed disparity values. A lower disparity indicates a closer object in the scene.

5. **Visualization:** The depth map is saved as 'depth1.png' and also displayed using OpenCV.

## Usage

To use this code for stereo depth map generation:

1. Ensure you have OpenCV installed in your Python environment.

2. Replace the paths to the left and right stereo images by modifying the lines:
   - `left = cv2.imread("imL.png", 0)`
   - `right = cv2.imread("imR.png", 0)`

3. Adjust the `kernel` and `max_offset` parameters to control the block matching window size and the maximum allowable disparity.

4. Run the script to generate and visualize the stereo depth map.

## Contribution and Modification

You are welcome to use, modify, and contribute to this code for your stereo depth map generation needs. If you find it valuable, consider giving this repository a star.

Feel free to explore the code, experiment with different images, and adapt it to your specific use cases. We hope this implementation helps you with your stereo vision projects.

## Acknowledgment

We acknowledge the open-source community, which makes it possible to share and collaborate on code like this. Special thanks to the contributors and maintainers of the libraries and tools used in this project.

---
