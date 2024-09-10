import time
import logging
import numpy as np
from ransac_algorithm import run_ransac
from utils import *


@log_execution_time
def run_reconstruction(
    relative_rotations: list, init_pair_dict: dict, K, pixel_threshold: float
):
    """
    Runs the 3D reconstruction process, computing absolute rotations and
    reconstructing initial 3D points from the input data.

    Args:
        relative_rotations (list): List of relative rotations between image pairs.
        init_pair_dict (dict): Dictionary with information for the initial image pair.
        K (np.ndarray): Camera intrinsic matrix.
        pixel_threshold (float): Threshold for pixel error in 3D point reconstruction.

    Returns:
        tuple: Contains the reconstructed 3D points (X0), absolute rotations, and inliers.
    """
    absolute_rotations = compute_absolute_rotations(relative_rotations)

    X0, inliers = reconstruct_initial_3D_points(
        init_pair_dict,
        K,
        absolute_rotations[init_pair_dict["init_pair_indices"][0]],
        pixel_threshold,
    )
    logging.info(
        f"Number of inliers for reconstruction of initial pair: {len(inliers)}"
    )
    return X0, absolute_rotations, inliers


@log_execution_time
def compute_absolute_rotations(relative_rotations: list) -> list:
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
def reconstruct_initial_3D_points(
    init_pair_dict: dict, K: list, R_init_1: list, pixel_threshold: float
):
    """
    Reconstructs initial 3D points from two image views using RANSAC and DLT triangulation.

    Args:
        init_pair_dict (dict): Dictionary containing matched points (x1, x2) from the initial image pair.
        K (np.ndarray): Camera intrinsic matrix.
        R_init_1 (np.ndarray): Initial rotation matrix for camera 1.
        pixel_threshold (float): Threshold for pixel error in RANSAC inlier determination.

    Returns:
        tuple: Contains the reconstructed 3D points (X0) and inliers.
    """
    x1 = init_pair_dict["x_init"][0]
    x2 = init_pair_dict["x_init"][1]
    R, T, inliers = run_ransac(K, x1, x2, pixel_threshold)

    P1 = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
    P2 = np.hstack((R, T))
    P = [P1, P2]
    xs = [normalize_K(K, x1), normalize_K(K, x2)]

    X0 = triangulate_3D_point_DLT(P, xs)
    X0 = X0[:3, :] / X0[3, :]
    X0 = np.dot(R_init_1.T, X0)

    return X0, inliers
