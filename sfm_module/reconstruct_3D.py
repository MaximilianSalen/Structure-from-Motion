import time
import logging
import numpy as np
from ransac_algorithm import run_ransac
from utils import triangulate_3D_point_DLT, log_execution_time


@log_execution_time
def run_reconstruction(
    relative_rotations, init_x1, init_x2, K, first_init_image_idx, pixel_threshold
):
    """
    Runs the 3D reconstruction process, computing absolute rotations and
    reconstructing initial 3D points from the input data.

    Args:
        relative_rotations (list): List of relative rotations between image pairs.
        init_x1 (np.ndarray): Keypoints from the first initial image.
        init_x2 (np.ndarray): Keypoints from the second initial image.
        K (np.ndarray): Camera intrinsic matrix.
        first_init_image_idx (int): Index of the first image in the initial pair.
        pixel_threshold (float): Threshold for pixel error in 3D point reconstruction.

    Returns:
        tuple: Contains the reconstructed 3D points (X0), absolute rotations, and inliers.
    """
    absolute_rotations = compute_absolute_rotations(relative_rotations)

    X0, inliers = reconstruct_initial_3D_points(
        init_x1, init_x2, K, absolute_rotations[first_init_image_idx], pixel_threshold
    )
    logging.info(
        f"Number of inliers for reconstruction of initial pair: {len(inliers)}"
    )
    return X0, absolute_rotations, inliers


@log_execution_time
def compute_absolute_rotations(relative_rotations: list):
    """
    Compute absolute rotations for each camera given relative rotations.

    :param relative_rotations: List of relative rotation matrices between consecutive cameras.
    :return: List of absolute rotation matrices for each camera.
    """

    # Initialize the first rotation matrix as identity (assuming the first camera is aligned with the global frame)
    absolute_rotations = [np.eye(3)]

    for relative_rotation in relative_rotations:
        # The next absolute rotation is the product of the current absolute rotation and the next relative rotation
        next_absolute_rotation = np.dot(absolute_rotations[-1], relative_rotation)
        absolute_rotations.append(next_absolute_rotation)

    return absolute_rotations


@log_execution_time
def reconstruct_initial_3D_points(x1, x2, K, R_i1, pixel_threshold):
    R, T, inliers = run_ransac(K, x1, x2, pixel_threshold)

    P1 = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
    P2 = np.hstack((R, T))
    P = [P1, P2]
    xs = [normalize_K(K, x1), normalize_K(K, x2)]

    X0 = triangulate_3D_point_DLT(P, xs)
    X0 = X0[:3, :] / X0[3, :]
    X0 = np.dot(R_i1.T, X0)

    return X0, inliers


def normalize_K(K, xs):
    return np.linalg.inv(K) @ xs
