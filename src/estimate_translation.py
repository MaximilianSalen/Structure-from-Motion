import numpy as np
from tqdm import tqdm
from scipy.linalg import lstsq
from .utils import *


@log_execution_time
def estimate_translation(
    K: list,
    desc_X: np.array,
    X0: np.array,
    absolute_rotations: list,
    img_paths: list,
    pixel_threshold: list,
) -> list:
    """
    Estimates translation vectors for each image using robust 2-point RANSAC.

    Args:
        K (np.ndarray): Camera intrinsic matrix.
        desc_X (np.ndarray): Descriptors of matched 3D points.
        X0 (np.ndarray): Reconstructed 3D points.
        absolute_rotations (list): Absolute rotation matrices for each image.
        img_paths (list): Paths to the input images.
        pixel_threshold (float): Threshold for RANSAC inlier selection.

    Returns:
        list: List of estimated translation vectors.
    """
    initial_Ts = []

    # Iterate over the number of images
    for i in tqdm(range(len(img_paths)), desc="Estimating translation vectors:"):
        X_corr, x_corr_norm = find_correspondences(img_paths[i], desc_X, X0, K)
        init_T = robust_estimate_T(
            x_corr_norm, X_corr, K, absolute_rotations[i], pixel_threshold
        )
        initial_Ts.append(init_T)

    return initial_Ts


def robust_estimate_T(x_norm, X, K, R, pixel_threshold):
    """
    Robustly estimates the translation vector using a 2-point RANSAC approach.

    Args:
        x_norm (np.ndarray): Normalized 2D points in the image (2xN).
        X (np.ndarray): Corresponding 3D points (3xN).
        K (np.ndarray): Camera calibration matrix.
        R (np.ndarray): Rotation matrix.
        pixel_threshold (float): Threshold for determining inliers.

    Returns:
        np.ndarray: Estimated translation vector.
    """
    num_iterations = 100000
    max_num_inliers = 0
    best_T = None
    threshold = 4 * pixel_threshold / K[0][0]

    for _ in range(num_iterations):
        # Randomly sample 2 correspondences
        inds = np.random.randint(0, x_norm.shape[1], size=2)
        x_sample = x_norm[:, inds]
        X_sample = X[:, inds]

        # Estimate T using the 2-point method
        T_estimated = estimate_T_2point(x_sample, X_sample, R)
        # Count inliers
        num_inliers = count_inliers(x_norm, X, R, T_estimated, threshold)

        if num_inliers > max_num_inliers:
            max_num_inliers = num_inliers
            best_T = T_estimated

    print(f"Number of inliers T: {max_num_inliers}")
    return best_T


def estimate_T_2point(x_norm, X, R):
    """
    Estimates the translation vector using a 2-point method.

    Args:
        x_norm (np.ndarray): Normalized 2D points.
        X (np.ndarray): Corresponding 3D points.
        R (np.ndarray): Rotation matrix.

    Returns:
        np.ndarray: Estimated translation vector.
    """

    A = []
    b = []
    for Xj, xij in zip(X.T, x_norm.T):
        x_skew = skew_symmetric_mat(xij)
        A.append(x_skew)
        b.append(-x_skew @ (R @ Xj))

    A = np.vstack(A)
    b = np.vstack(b).reshape(-1)

    # Solve for T using least squares
    T, _, _, _ = lstsq(A, b)

    return T.flatten()


def count_inliers(x_norm, X, R, T_estimated, threshold):
    """
    Counts the number of inliers by projecting 3D points to the image and comparing with 2D points.

    Args:
        x_norm (np.ndarray): Normalized 2D points.
        X (np.ndarray): 3D points.
        R (np.ndarray): Rotation matrix.
        T_estimated (np.ndarray): Estimated translation vector.
        threshold (float): Distance threshold for inliers.

    Returns:
        int: Number of inliers.
    """
    x_projected = R @ X + T_estimated[:, np.newaxis]
    x_projected /= x_projected[2, :]  # Normalize to 2D

    distances = np.linalg.norm(x_projected[:2, :] - x_norm[:2, :], axis=0)
    inliers = np.sum(distances < threshold)

    return inliers
