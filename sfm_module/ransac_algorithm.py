import numpy as np
import time
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from homography_to_RT import homography_to_RT
from essential_to_RT import essential_to_RT


def estimate_R(K, x_pairs, pixel_threshold):
    """
    Runs the RANSAC algorithm on a series of image pairs to estimate rotation and translation matrices.

    Args:
        K (np.ndarray): Camera intrinsic matrix.
        x_pairs (list): List of corresponding keypoints between image pairs.
        pixel_threshold (float): Pixel distance threshold for RANSAC inlier selection.

    Returns:
        tuple: Contains lists of RT (rotation and translation), R (rotation), and T (translation) matrices.
    """

    R_list = []

    for i in tqdm(range(len(x_pairs) // 2), desc="Running RANSAC"):
        start_time = time.time()
        x1 = x_pairs[2 * i]
        x2 = x_pairs[2 * i + 1]

        logging.info(f"Processing pair {i+1}/{len(x_pairs) // 2}")

        # Normalize keypoints
        x1_norm = normalize_K(K, x1)
        x2_norm = normalize_K(K, x2)

        # Estimate essential matrix with RANSAC and get rotation, translation, and inliers
        i_R, _, inliers = modified_estimate_E_robust(
            K, x1_norm, x2_norm, pixel_threshold
        )
        elapsed_time = time.time() - start_time
        logging.info(
            f"RANSAC completed for pair {i+1} in {elapsed_time:.2f} seconds with {len(inliers)} inliers"
        )
        R_list.append(i_R)

    return R_list


def run_ransac(K, x1, x2, pixel_threshold):
    x1_norm = normalize_K(K, x1)
    x2_norm = normalize_K(K, x2)
    best_R, best_T, inliers = modified_estimate_E_robust(
        K, x1_norm, x2_norm, pixel_threshold
    )
    return best_R, best_T, inliers


def normalize_K(K, xs):
    return np.linalg.inv(K) @ xs


def enforce_essential(E):
    U, S, Vt = np.linalg.svd(E)
    S[2] = 0  # Set the smallest singular value to zero
    S[0] = S[1] = 1  # Set the non-zero singular values to 1
    return U @ np.diag(S) @ Vt


def epipolar_errors(F, x1, x2):
    l = F @ x1
    # Normalization of the lines
    l /= np.sqrt(np.repeat((l[0, :] ** 2 + l[1, :] ** 2)[np.newaxis, :], 3, axis=0))
    d = np.abs(np.sum(l * x2, axis=0))
    return d


def estimate_F_DLT(x1s, x2s):
    # Ensure points_2D are in homogeneous coordinates if necessary
    if x1s.shape[0] == 2:
        x1s = np.vstack([x1s, np.ones((1, x1s.shape[1]))])

    if x2s.shape[0] == 2:
        x2s = np.vstack([x2s, np.ones((1, x2s.shape[1]))])

    # Number of points
    num_points = x1s.shape[1]

    # Build the matrix A for DLT
    A = np.zeros((num_points, 9))
    for i in range(num_points):
        x1, y1, z1 = x1s[:, i]
        x2, y2, z2 = x2s[:, i]
        A[i] = [
            x1 * x2,
            x1 * y2,
            x1 * z2,
            y1 * x2,
            y1 * y2,
            y1 * z2,
            z1 * x2,
            z1 * y2,
            z1 * z2,
        ]

    # Apply Singular Value Decomposition (SVD)
    U, S, Vh = np.linalg.svd(A)

    v = Vh[-1]

    # The solution is the last column of V (or Vh.T)
    F = v.reshape((3, 3)).T

    return F


def estimate_H_DLT(x1s, x2s):
    # Construct matrix A
    A = []
    for (x1, y1, _), (x2, y2, _) in zip(x1s.T, x2s.T):
        A.append([-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2])
        A.append([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])

    A = np.array(A)

    # Apply Singular Value Decomposition (SVD)
    U, S, Vh = np.linalg.svd(A)

    H = Vh[-1].reshape((3, 3))

    # Normalize: H[2, 2] = 1
    return H / H[2, 2]


def get_inlier_mask_H(H, x1s, x2s, threshold):
    # Ensure points_2D are in homogeneous coordinates if necessary
    if x1s.shape[0] == 2:
        x1s = np.vstack([x1s, np.ones((1, x1s.shape[1]))])

    if x2s.shape[0] == 2:
        x2s = np.vstack([x2s, np.ones((1, x2s.shape[1]))])

    # Applying Homography to points in the first image
    transformed_points = H @ x1s
    transformed_points /= transformed_points[
        2, :
    ]  # Normalize to make the third coordinate 1

    # Distance between transformed points and points in the second image
    distances = np.sqrt(np.sum((x2s[:2, :] - transformed_points[:2, :]) ** 2, axis=0))
    mask = distances < threshold

    return mask


def skew_symmetric_mat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def modified_estimate_E_robust(K, x1, x2, pixel_threshold):

    s_E = 8
    s_H = 4
    alpha = 0.95
    epsilon_E = 0.1
    epsilon_H = 0.1
    best_num_inliers_E = 0
    best_num_inliers_H = 0
    err_threshold = pixel_threshold / K[0][0]

    # initial nr of iterations for E and H
    E_iters = np.abs(np.log(1 - alpha) / np.log(1 - epsilon_E**s_E))
    H_iters = np.abs(np.log(1 - alpha) / np.log(1 - epsilon_H**s_H))

    iterations = 0
    while iterations < max(E_iters, H_iters):

        ###### Estimate E ######
        inds_E = np.random.randint(0, x1.shape[1], size=s_E)
        E_adjusted = enforce_essential(estimate_F_DLT(x1[:, inds_E], x2[:, inds_E]))
        inlier_mask = (
            epipolar_errors(E_adjusted, x1, x2) ** 2
            + epipolar_errors(E_adjusted.T, x2, x1) ** 2
        ) / 2 < err_threshold**2
        num_inliers_E = np.sum(inlier_mask)

        # Update E if there is more inliers than before
        if num_inliers_E > best_num_inliers_E:
            best_num_inliers_E = num_inliers_E
            # Get R, T (and also number of points infront of camera which is not needed here, therefore "_") from E and set it to best R and T
            R_best, T_best, _ = essential_to_RT(E_adjusted, K, x1, x2)
            epsilon_E = best_num_inliers_E / x1.shape[1]
            E_iters = np.abs(np.log(1 - alpha) / np.log(1 - epsilon_E**s_E))
            inliers = inlier_mask

        ###### Estimate H ######
        inds_H = np.random.randint(0, x1.shape[1], size=s_H)
        H = estimate_H_DLT(x1[:, inds_H], x2[:, inds_H])
        inlier_mask = get_inlier_mask_H(
            H, x1, x2, err_threshold * 3
        )  # it is given that the error threshold should be mult. by 3 for H
        num_inliers_H = np.sum(inlier_mask)

        if num_inliers_H > best_num_inliers_H:
            best_num_inliers_H = num_inliers_H

            # Get RT sols from H
            R_a, T_a, R_b, T_b = homography_to_RT(H, x1, x2)
            E_a = skew_symmetric_mat(T_a) @ R_a
            E_b = skew_symmetric_mat(T_b) @ R_b

            # Mask the inliers that satisfies the epipolar constraint
            inlier_mask_a = (
                epipolar_errors(E_a, x1, x2) ** 2 + epipolar_errors(E_a.T, x2, x1) ** 2
            ) / 2 < err_threshold**2
            inlier_mask_b = (
                epipolar_errors(E_b, x1, x2) ** 2 + epipolar_errors(E_b.T, x2, x1) ** 2
            ) / 2 < err_threshold**2

            # Get R, T and number of inliers that are infront of camera from E and set it to best R and T. Note that the points also satisfy the epipolar constraint due to the mask
            R_best_a, T_best_a, num_inliers_E_a = essential_to_RT(
                E_a, K, x1[:, inlier_mask_a], x2[:, inlier_mask_a]
            )
            R_best_b, T_best_b, num_inliers_E_b = essential_to_RT(
                E_b, K, x1[:, inlier_mask_b], x2[:, inlier_mask_b]
            )

            if num_inliers_E_a > num_inliers_E_b:
                if num_inliers_E_a > num_inliers_E:
                    R_best = R_best_a
                    T_best = T_best_a
                    best_num_inliers_E = num_inliers_E_a
                    epsilon_H = best_num_inliers_H / x1.shape[1]
                    epsilon_E = best_num_inliers_E / x1.shape[1]
                    H_iters = np.abs(np.log(1 - alpha) / np.log(1 - epsilon_H**s_H))
                    E_iters = np.abs(np.log(1 - alpha) / np.log(1 - epsilon_E**s_E))
                    inliers = inlier_mask_a

            else:
                if num_inliers_E_b > num_inliers_E:
                    R_best = R_best_b
                    T_best = T_best_b
                    best_num_inliers_E = num_inliers_E_b
                    epsilon_H = best_num_inliers_H / x1.shape[1]
                    epsilon_E = best_num_inliers_E / x1.shape[1]
                    H_iters = np.abs(np.log(1 - alpha) / np.log(1 - epsilon_H**s_H))
                    E_iters = np.abs(np.log(1 - alpha) / np.log(1 - epsilon_E**s_E))
                    inliers = inlier_mask_b

        iterations += 1

    return R_best, np.reshape(T_best, (3, 1)), inliers


# def modified_estimate_E_robust(K, x1, x2, pixel_threshold):

#     s_E = 8
#     s_H = 4
#     alpha = 0.95
#     epsilon_E = 0.1
#     epsilon_H = 0.1
#     best_num_inliers_E = 0
#     best_num_inliers_H = 0
#     err_threshold = pixel_threshold / K[0][0]

#     # initial nr of iterations for E and H
#     E_iters = np.abs(np.log(1-alpha)/np.log(1-epsilon_E**s_E))
#     H_iters = np.abs(np.log(1-alpha)/np.log(1-epsilon_H**s_H))

#     for i in range(100000):
#       ###### Estimate E ######
#         inds_E = np.random.randint(0, x1.shape[1], size=s_E)
#         E_adjusted = enforce_essential(estimate_F_DLT(x1[:, inds_E], x2[:, inds_E]))
#         inlier_mask = (epipolar_errors(E_adjusted, x1, x2)**2 + epipolar_errors(E_adjusted.T, x2, x1)**2) / 2 < err_threshold**2
#         num_inliers_E = np.sum(inlier_mask)

#         # Update E if there is more inliers than before
#         if num_inliers_E > best_num_inliers_E:
#             best_num_inliers_E = num_inliers_E
#             # Get R, T (and also number of points infront of camera which is not needed here, therefore "_") from E and set it to best R and T
#             R_best, T_best, _ = essential_to_RT(E_adjusted, K, x1, x2)
#             epsilon_E = best_num_inliers_E / x1.shape[1]
#             E_iters = np.abs(np.log(1-alpha)/np.log(1-epsilon_E**s_E))

#     return R_best, np.reshape(T_best, (3,1))
