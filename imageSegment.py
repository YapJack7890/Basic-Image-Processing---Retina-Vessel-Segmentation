# -*- coding: utf-8 -*-
"""
imageSegment.py

YOUR WORKING FUNCTION

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

input_dir = 'dataset/test'
output_dir = 'dataset/output'
gt_dir = 'dataset/groundtruth'

# you are allowed to import other Python packages above
##########################
def segmentImage(img):
    # Inputs
    # img: Input image, a 3D numpy array of row*col*3 in BGR format
    #
    # Output
    # outImg: segmentation image
    #
    #########################################################################
    # ADD YOUR CODE BELOW THIS LINE
    # Step 1: Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Intensity Normalization
    # Calculate mean and standard deviation
    mean = np.mean(gray)
    std_dev = np.std(gray)
    # Apply the 68-95-99.7 rule
    low_2sigma = mean - 2 * std_dev
    high_2sigma = mean + 2 * std_dev
    # Clip intensities to ±2σ
    image_clipped = np.clip(gray, low_2sigma, high_2sigma)
    # Normalize intensities
    image_normalized = ((image_clipped - low_2sigma) / (high_2sigma - low_2sigma) * 255).astype('uint8')

    # Step 3: Smoothing (Gaussian Blur)
    sigma1 = 1.0  # Smaller sigma
    sigma2 = 2.0  # Larger sigma
    gaussian1 = cv2.GaussianBlur(image_normalized, (3, 3), sigma1) # kernel 3x3
    gaussian2 = cv2.GaussianBlur(image_normalized, (5, 5), sigma2) # kernel 5x5

    # Step 4: Edge enhancement (DoG)
    dog = gaussian1 - gaussian2
    
    # Step 5: Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1=200, threshold2=210)
    kernel = np.ones((3, 3), np.uint8)  # 3x3 square kernel
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    dog = cv2.subtract(dog, dilated_edges)
    
    # Step 6: Normalize the DoG output for better contrast
    dog_normalized = cv2.normalize(dog, None, 0, 1, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Step 7: Median Blur
    cleaned_mask = cv2.medianBlur(dog_normalized, 3)

    # Step 8: Connected Components Analysis
    _, labels = cv2.connectedComponents(cleaned_mask)
    component_sizes = np.bincount(labels.flatten())  # Count pixel frequencies for each label
    valid_labels = np.where(component_sizes >= 200)[0] # Create a mask of valid components
    valid_labels = valid_labels[valid_labels != 0]  # Exclude background (label 0)
    # Use NumPy to directly filter components
    cleaned_mask_removed = np.isin(labels, valid_labels).astype(np.uint8) * 255

    # Step 6: Post-processing - Morphological closing
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask_removed = cv2.morphologyEx(cleaned_mask_removed, cv2.MORPH_CLOSE, kernel,  iterations=1)
    
    # Step 9: Normalize the output to the range [0, 1]
    outImg = cv2.normalize(cleaned_mask_removed, None, 0, 1, cv2.NORM_MINMAX)

    # END OF YOUR CODE
    #########################################################################
    return outImg
