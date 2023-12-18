# Bronte Sihan Li, Cole Crescas, Karan Shah
# 2023-09-23
# CS 7180
# This is largely based on the work of: https://github.com/Utkarsh-Deshmukh/Single-Image-Dehazing-Python/

"""
Simple implementation of dehazing algorithm from the paper:
"Efficient Image Dehazing with Boundary Constraint and Contextual Regularization"
We use this on our own dataset to compare with the DehazeFormer model from a more qualitative perspective.

@INPROCEEDINGS{6751186, author={G. Meng and Y. Wang and J. Duan and S. Xiang and C. Pan},
booktitle={IEEE International Conference on Computer Vision},
title={Efficient Image Dehazing with Boundary Constraint and Contextual Regularization},
year={2013}, volume={}, number={}, pages={617-624}, month={Dec},}
"""

"""
Libraries needed: 

1.numpy==1.19.0

2.opencv-python

3.scipy

Theory:
Airlight estimation
Calculating boundary constraints
Estimate and refine Transmission
Perform Dehazing using the estimated Airlight and Transmission
"""

import image_dehazer
import cv2
import os

# Our specific path to the dataset locally and results section
haze_dir = 'data/a2i2/UAV-train/paired_dehaze/images/hazy/'
result_dir = 'results/image_dehazer/'
os.makedirs(result_dir, exist_ok=True)
#Three images to use
images_to_sample = [
    '060.png',
    '074.png',
    '080.png',
]

for image in images_to_sample:
    hazy_img = cv2.imread(haze_dir + image)
    hazy_corrected, haze_map = image_dehazer.remove_haze(hazy_img)
    cv2.imshow('input', hazy_img)
    cv2.imshow('hazy_corrected', hazy_corrected)
    cv2.imshow('haze_map', haze_map)
    # cv2.waitKey(0)
    cv2.imwrite(result_dir + image, hazy_corrected)

### user controllable parameters (with their default values), we will adjust these to work best on real world data:
airlightEstimation_windowSze = 15
boundaryConstraint_windowSze = 3
C0 = 20
C1 = 300
regularize_lambda = 0.1
sigma = 0.5
delta = 0.85
showHazeTransmissionMap = True
