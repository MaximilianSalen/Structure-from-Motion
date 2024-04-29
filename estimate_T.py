"""
Computer Vision
EEN020
Project
2023-12-30

Estimate the translation T robustly from 2D-3D correspondences xi <-> Xi

Authors:
        Maximilian Salén
        Axel Qvarnström
"""

import numpy as np
import time
import cv2
from scipy.linalg import lstsq
from tqdm import trange

def get_T(x_norm, X, K, R, inlier_threshold):

    # Get the correspondences
    start_time = time.time()
    #X, x_norm = get_correspondences(image_name, desc_X, X0, K)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time Get correspondences: {elapsed_time} seconds")

    start_time = time.time()
    T = estimate_T_robust(x_norm, X, K, R, inlier_threshold)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time Estimate T Robust: {elapsed_time} seconds")

    return T


def estimate_T_robust(x, X, K, R, inlier_threshold):
    """
    Robustly estimates the translation vector T from 2D-3D point correspondences using a 2-point method and RANSAC.

    :param x: 2D points in the image (2xN)
    :param X: Corresponding 3D points (3xN)
    :param K: Camera calibration matrix
    :param R: Rotation matrix
    :param inlier_threshold: Threshold for determining inliers
    :return: Estimated translation vector T
    """
    num_iterations = 100000
    max_num_inliers = 0
    best_T = None
    inlier_threshold = 4 * inlier_threshold / K[0][0] 
    

    for _ in range(num_iterations):
        inds = np.random.randint(0, x.shape[1], size=2)
        x_sample = x[:, inds]
        X_sample = X[:, inds]

        # estimate T using 2-point method
        T_estimated = estimate_T_2point(x_sample, X_sample, K, R)
        # count inliers
        num_inliers = get_inliers(x, X, K, R, T_estimated, inlier_threshold)
        
        if num_inliers > max_num_inliers:
            max_num_inliers = num_inliers
            best_T = T_estimated

    print('number of inliers T: ', max_num_inliers)
    return best_T


def get_correspondences(image_name, desc_X, X0, K):
    image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(descriptors, desc_X.T, k=2)

    # Apply ratio test to filter good matches
    matches_2d_3d = []
    for m, n in raw_matches:
            matches_2d_3d.append(m)

    x = np.float32([keypoints[m.queryIdx].pt for m in matches_2d_3d]).T
    X = np.float32([X0[:, m.trainIdx] for m in matches_2d_3d]).T
    x_norm = normalize_K(K, cartesian_to_homogeneous(x))
    return X, x_norm



###### Helping functions #######

def normalize_K(K, xs):
    return np.linalg.inv(K) @ xs

def cartesian_to_homogeneous(cartesian_points):
    # Add a row of ones at the bottom of the cartesian_points matrix
    homogeneous_points = np.vstack(
        (cartesian_points, np.ones((1, cartesian_points.shape[1])))
    )
    return homogeneous_points

def estimate_T_2point(x_norm, X, K, R):

    # # Rotate 3D points
    # rotated_X = np.dot(R, X)

    # A = np.zeros((2 * X.shape[1], 3))
    # b = np.zeros(2 * X.shape[1])
    # for i in range(X.shape[1]):
    #     A[2*i:2*i+2, :] = rotated_X[:, i:i+1].T
    #     b[2*i:2*i+2] = x_norm[:2, i]


    A = []
    b = []
    for Xj, xij in zip(X.T, x_norm.T):
        x_skew = skew_symmetric_mat(xij)
        A.append(x_skew)
        b.append(-x_skew @ (R @ Xj))

    A = np.vstack(A)
    b = np.vstack(b).reshape(-1)

    # solve for T
    T, _, _, _ = lstsq(A, b)

    return T.flatten()

def get_inliers(x_norm, X, K, R, T_estimated, threshold):
    x_projected = R @ X + T_estimated[:, np.newaxis]

    # project the points to the 2D plane
    # x_projected = K @ X_projected
    x_projected /= x_projected[2, :] # Normalize to 2D

    distances = np.linalg.norm(x_projected - x_norm, axis=0)
    inliers = np.sum(distances < threshold)

    return inliers


def skew_symmetric_mat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


# def estimate_T_robust(x, X, K, R, inlier_threshold):
#     """
#     Robustly estimates the translation vector T from 2D-3D point correspondences using a 2-point method and RANSAC.

#     :param x: 2D points in the image (2xN)
#     :param X: Corresponding 3D points (3xN)
#     :param K: Camera calibration matrix
#     :param R: Rotation matrix
#     :param inlier_threshold: Threshold for determining inliers
#     :return: Estimated translation vector T
#     """
   
#     max_num_inliers = 0
#     best_T = None
#     inlier_threshold = 3 * inlier_threshold / K[0][0] 
#     alpha = 0.95
#     epsilon_E = 0.1      # given threshold for estimating T
#     s_E = 2
#     E_iters = np.abs(np.log(1-alpha)/np.log(1-epsilon_E**s_E))

#     iterations = 0
#     while iterations < E_iters:
#         inds = np.random.randint(0, x.shape[1], size=2)
#         x_sample = x[:, inds]
#         X_sample = X[:, inds]

#         # estimate T using 2-point method
#         T_estimated = estimate_T_2point(x_sample, X_sample, K, R)
#         # count inliers
#         num_inliers = get_inliers(x, X, K, R, T_estimated, inlier_threshold)
        
#         if num_inliers > max_num_inliers:
#             max_num_inliers = num_inliers
#             best_T = T_estimated
#             epsilon_E = max_num_inliers / x.shape[1]
#             E_iters = np.abs(np.log(1-alpha)/np.log(1-epsilon_E**s_E))
        
#         iterations += 1

#     return best_T