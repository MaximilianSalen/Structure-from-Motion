import logging
import numpy as np
from utils import *
from ransac_algorithm import run_ransac


@log_execution_time
def run_reconstruction(
    relative_rotations: list, init_pair_dict: dict, K: list, pixel_threshold: float
):
    """
    Reconstructs initial 3D points from two image views using RANSAC and DLT triangulation.

    Args:
        relative_rotations (list): List of relative rotation matrices between consecutive cameras.
        init_pair_dict (dict): Dictionary containing matched points (x1, x2) from the initial image pair.
        K (list): Camera intrinsic matrix.
        pixel_threshold (float): Threshold for pixel error in RANSAC inlier determination.

    Returns:
        tuple:
            - X0 (np.ndarray): Reconstructed 3D points in camera 1's coordinate frame.
            - absolute_rotations (list): Absolute rotation matrices for each camera.
            - inliers (list): Inliers detected by RANSAC.
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
    Computes absolute rotations for each camera given relative rotations.

    Args:
        relative_rotations (list): List of relative rotation matrices between consecutive cameras.

    Returns:
        list: Absolute rotation matrices for each camera.
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
        K (list): Camera intrinsic matrix.
        R_init_1 (np.ndarray): Rotation matrix for first image in initial pair.
        pixel_threshold (float): Threshold for pixel error in RANSAC inlier determination.

    Returns:
        tuple:
            - X0 (np.ndarray): Reconstructed 3D points in camera 1's coordinate frame.
            - inliers (list): Inliers detected by RANSAC.
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
