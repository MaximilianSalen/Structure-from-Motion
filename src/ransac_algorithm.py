import numpy as np
import time
import logging
import scipy
from tqdm import tqdm
from .chirality_check import perform_chirality_check
from .utils import *


def estimate_R(K: list, x_pairs: list, pixel_threshold: float) -> list:
    """
    Runs the RANSAC algorithm on a series of image pairs to estimate rotation matrices.

    Args:
        K (np.ndarray): Camera intrinsic matrix.
        x_pairs (list): List of corresponding keypoints between image pairs.
        pixel_threshold (float): Pixel distance threshold for RANSAC inlier selection.

    Returns:
        list: Rotation matrices (R) estimated from each image pair.
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
        i_R, _, inliers = estimate_E_robust(K, x1_norm, x2_norm, pixel_threshold)
        elapsed_time = time.time() - start_time
        logging.info(
            f"RANSAC completed for pair {i+1} in {elapsed_time:.2f} seconds with {len(inliers)} inliers"
        )
        R_list.append(i_R)

    return R_list


def run_ransac(K: list, x1: np.array, x2: np.array, pixel_threshold: float):
    """
    Main function to run ransac algorithm and estimate the essential matrix robustly.
    """
    x1_norm = normalize_K(K, x1)
    x2_norm = normalize_K(K, x2)
    best_R, best_T, inliers = estimate_E_robust(K, x1_norm, x2_norm, pixel_threshold)
    return best_R, best_T, inliers


def estimate_E_robust(K: list, x1: np.array, x2: np.array, pixel_threshold: float):
    """
    Estimates the essential matrix robustly using RANSAC and iterates over both essential matrix (E)
    and homography (H) estimations to find the best solution. Returns the best rotation and translation matrices.

    Args:
        K (list): Camera intrinsic matrix.
        x1 (np.ndarray): Keypoints from the first image.
        x2 (np.ndarray): Keypoints from the second image.
        pixel_threshold (float): Pixel distance threshold for RANSAC inlier determination.

    Returns:
        tuple:
            - R_best (np.ndarray): Best estimated rotation matrix.
            - T_best (np.ndarray): Best estimated translation matrix.
            - inliers (np.ndarray): Boolean mask of inliers detected by RANSAC.
    """
    # Parameters for RANSAC iterations
    s_E = 8  # Minimum points required for E estimation
    s_H = 4  # Minimum points required for H estimation
    alpha = 0.95  # Confidence level
    epsilon_E = 0.1  # Initial inlier ratio estimate for E
    epsilon_H = 0.1  # Initial inlier ratio estimate for H
    best_num_inliers_E = 0  # Best inliers count for E
    best_num_inliers_H = 0  # Best inliers count for H
    err_threshold = (
        pixel_threshold / K[0][0]
    )  # Normalized error threshold based on camera matrix

    # Calculate initial number of iterations for E and H
    E_iters = np.abs(np.log(1 - alpha) / np.log(1 - epsilon_E**s_E))
    H_iters = np.abs(np.log(1 - alpha) / np.log(1 - epsilon_H**s_H))

    iterations = 0
    while iterations < max(E_iters, H_iters):

        ###### Estimate E ######
        inds_E = np.random.randint(0, x1.shape[1], size=s_E)
        E_adjusted = enforce_essential(estimate_F_DLT(x1[:, inds_E], x2[:, inds_E]))

        # Compute inliers based on epipolar constraint
        inlier_mask = (
            epipolar_errors(E_adjusted, x1, x2) ** 2
            + epipolar_errors(E_adjusted.T, x2, x1) ** 2
        ) / 2 < err_threshold**2
        num_inliers_E = np.sum(inlier_mask)

        # Update E if there is more inliers than before
        if num_inliers_E > best_num_inliers_E:
            best_num_inliers_E = num_inliers_E

            # Get R, T (and also number of points infront of camera which is not needed here, therefore "_")
            # from E and set it to best R and T
            R_best, T_best, _ = essential_to_RT(E_adjusted, K, x1, x2)
            epsilon_E = best_num_inliers_E / x1.shape[1]
            E_iters = np.abs(np.log(1 - alpha) / np.log(1 - epsilon_E**s_E))
            inliers = inlier_mask

        ###### Estimate H ######
        inds_H = np.random.randint(0, x1.shape[1], size=s_H)
        H = estimate_H_DLT(x1[:, inds_H], x2[:, inds_H])

        # Compute inliers for H, with a higher threshold (multiplied by 3)
        inlier_mask = get_inlier_mask_H(H, x1, x2, err_threshold * 3)
        num_inliers_H = np.sum(inlier_mask)

        # Update the best H if more inliers are found
        if num_inliers_H > best_num_inliers_H:
            best_num_inliers_H = num_inliers_H

            # Convert H to possible rotation and translation (R, T) pairs
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

            # Get R, T and number of inliers that are infront of camera from E and set it to best R and T.
            # Note that the points also satisfy the epipolar constraint due to the mask
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


def enforce_essential(E: np.array) -> np.array:
    """
    Enforces the essential matrix constraints by adjusting its singular values.
    Specifically, it ensures that the matrix has two equal singular values and the third is zero.

    Args:
        E (np.ndarray): Estimated essential matrix.

    Returns:
        np.ndarray: Adjusted essential matrix with enforced constraints.
    """
    U, S, Vt = np.linalg.svd(E)
    S[2] = 0  # Set the smallest singular value to zero
    S[0] = S[1] = 1  # Set the non-zero singular values to 1
    return U @ np.diag(S) @ Vt


def epipolar_errors(F: np.array, x1: np.array, x2: np.array):
    """
    Computes the epipolar errors between two sets of points given the fundamental matrix.

    Args:
        F (np.ndarray): Fundamental matrix relating the two views.
        x1 (np.ndarray): Points from the first image in homogeneous coordinates (3xN).
        x2 (np.ndarray): Corresponding points from the second image in homogeneous coordinates (3xN).

    Returns:
        np.ndarray: Epipolar errors for each corresponding point pair.
    """
    # Compute the epipolar lines for points in x1 in the second image
    l = F @ x1

    # Normalization of the epipolar lines
    l /= np.sqrt(np.repeat((l[0, :] ** 2 + l[1, :] ** 2)[np.newaxis, :], 3, axis=0))

    # Compute the absolute distance between points in x2 and the corresponding epipolar lines
    d = np.abs(np.sum(l * x2, axis=0))
    return d


def estimate_F_DLT(x1s: np.array, x2s: np.array):
    """
    Estimates the fundamental matrix using the Direct Linear Transformation (DLT) method.

    Args:
        x1s (np.ndarray): Points from the first image in homogeneous coordinates (3xN) or non-homogeneous (2xN).
        x2s (np.ndarray): Corresponding points from the second image in homogeneous coordinates (3xN) or non-homogeneous (2xN).

    Returns:
        np.ndarray: Estimated fundamental matrix (3x3).
    """

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


def estimate_H_DLT(x1s: np.array, x2s: np.array) -> np.array:
    """
    Estimates the homography matrix using the Direct Linear Transformation (DLT) method.

    Args:
        x1s (np.ndarray): Points from the first image in homogeneous coordinates (3xN).
        x2s (np.ndarray): Corresponding points from the second image in homogeneous coordinates (3xN).

    Returns:
        np.ndarray: Estimated homography matrix (3x3).
    """

    # Construct matrix A
    A = []
    for (x1, y1, _), (x2, y2, _) in zip(x1s.T, x2s.T):
        A.append([-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2])
        A.append([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])

    A = np.array(A)

    # Apply Singular Value Decomposition (SVD)
    U, S, Vh = np.linalg.svd(A)

    # The solution is the last column of Vh, reshaped as the homography matrix H
    H = Vh[-1].reshape((3, 3))

    # Normalize: H[2, 2] = 1
    return H / H[2, 2]


def get_inlier_mask_H(
    H: np.array, x1s: np.array, x2s: np.array, threshold: float
) -> np.array:
    """
    Computes an inlier mask for homography by transforming points from one image to the other and checking the distance to the corresponding points.

    Args:
        H (np.ndarray): Homography matrix (3x3).
        x1s (np.ndarray): Points from the first image in homogeneous coordinates (3xN) or non-homogeneous (2xN).
        x2s (np.ndarray): Corresponding points from the second image in homogeneous coordinates (3xN) or non-homogeneous (2xN).
        threshold (float): Distance threshold for determining inliers.

    Returns:
        np.ndarray: Boolean mask indicating which point correspondences are inliers.
    """

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

    # Calculate the Euclidean distance between transformed points and the actual points in the second image
    distances = np.sqrt(np.sum((x2s[:2, :] - transformed_points[:2, :]) ** 2, axis=0))
    mask = distances < threshold

    return mask


def homography_to_RT(H: np.array, x1: np.array, x2: np.array):
    """
    Decomposes a homography matrix into possible rotation (R) and translation (T) pairs.

    Args:
        H (np.ndarray): Homography matrix (3x3).
        x1 (np.ndarray): Keypoints from the first image.
        x2 (np.ndarray): Corresponding keypoints from the second image.

    Returns:
        tuple:
            R1 (np.ndarray): First possible rotation matrix.
            t1 (np.ndarray): First possible translation vector.
            R2 (np.ndarray): Second possible rotation matrix.
            t2 (np.ndarray): Second possible translation vector.
    """

    def unitize(a, b):
        denom = 1.0 / np.sqrt(a**2 + b**2)
        return a * denom, b * denom

    # Ensure that the homography matrix H has the correct sign
    N = x1.shape[1]
    if x1.shape[0] != 3:
        x1 = np.vstack([x1, np.ones((1, N))])
    if x2.shape[0] != 3:
        x2 = np.vstack([x2, np.ones((1, N))])
    positives = np.sum((np.sum(x2 * (np.dot(H, x1)), axis=0)) > 0)
    if positives < N / 2:
        H *= -1

    # Perform SVD on the homography matrix
    U, S, Vt = np.linalg.svd(H)
    s1 = S[0] / S[1]
    s3 = S[2] / S[1]

    # Compute parameters for the rotation and translation matrices
    a1 = np.sqrt(1 - s3**2)
    b1 = np.sqrt(s1**2 - 1)
    a, b = unitize(a1, b1)
    c, d = unitize(1 + s1 * s3, a1 * b1)
    e, f = unitize(-b / s1, -a / s3)

    # Extract vectors from the Vt matrix
    v1 = Vt.T[:, 0]
    v3 = Vt.T[:, 2]

    # Compute possible normal vectors
    n1 = b * v1 - a * v3
    n2 = b * v1 + a * v3

    # Compute the two possible rotation matrices
    R1 = U @ np.array([[c, 0, d], [0, 1, 0], [-d, 0, c]]) @ Vt
    R2 = U @ np.array([[c, 0, -d], [0, 1, 0], [d, 0, c]]) @ Vt

    # Compute the two possible translation vectors
    t1 = e * v1 + f * v3
    t2 = e * v1 - f * v3

    # Adjust signs of normal vectors and translations
    if n1[2] < 0:
        t1 = -t1
        n1 = -n1
    if n2[2] < 0:
        t2 = -t2
        n2 = -n2

    # Transform translations to Hartley and Zisserman convention
    t1 = np.dot(R1, t1)
    t2 = np.dot(R2, t2)

    return R1, t1, R2, t2


def essential_to_RT(E: np.array, K: list, x1: np.array, x2: np.array):
    """
    Decomposes the essential matrix into possible rotation (R) and translation (T) matrices.

    Args:
        E (np.ndarray): Essential matrix.
        K (list): Camera intrinsic matrix.
        x1 (np.ndarray): Keypoints from the first image.
        x2 (np.ndarray): Corresponding keypoints from the second image.

    Returns:
        tuple:
            best_R (np.ndarray): Best estimated rotation matrix.
            best_T (np.ndarray): Best estimated translation vector.
            num_points_infront_of_cam (int): Number of points in front of the camera.
    """
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # Perform SVD on the essential matrix
    U, S, Vt = scipy.linalg.svd(E)

    # Ensure the determinant of U and Vt is positive
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Vt = -Vt

    # Compute four possible projection matrices
    P1 = np.hstack((U @ W @ Vt, U[:, -1].reshape(-1, 1)))
    P2 = np.hstack((U @ W @ Vt, -U[:, -1].reshape(-1, 1)))
    P3 = np.hstack((U @ W.T @ Vt, U[:, -1].reshape(-1, 1)))
    P4 = np.hstack((U @ W.T @ Vt, -U[:, -1].reshape(-1, 1)))

    # Perform chirality check to find the best projection matrix
    best_P, num_points_infront_of_cam = perform_chirality_check(
        [P1, P2, P3, P4], K, x1, x2
    )

    # Extract the best rotation and translation matrices
    best_R = best_P[:, :3]
    best_T = best_P[:, 3]

    return best_R, best_T, num_points_infront_of_cam
