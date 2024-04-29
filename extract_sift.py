"""
Computer Vision
EEN020
Project
2023-12-11

Extracts SIFT features

Authors:
        Maximilian Salén
        Axel Qvarnström
"""

import cv2
import numpy as np
from auxiliary import cartesian_to_homogeneous

def execute_sift_extraction(image_name1, image_name2):
    # Load images
    im1 = cv2.imread(image_name1)
    im2 = cv2.imread(image_name2)
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect SIFT features and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Match descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good_matches.append(m)

    # Extract coordinates of matched keypoints
    x1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).T
    x2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).T

    x1 = cartesian_to_homogeneous(x1)
    x2 = cartesian_to_homogeneous(x2)

    # Save descriptors
    desc_X1 = np.array([descriptors1[m.queryIdx] for m in good_matches]).T
    
    return x1, x2, desc_X1