import numpy as np
from .utils import triangulate_3D_point_DLT


def perform_chirality_check(P2s: list, x1: np.array, x2: np.array):
    """
    Performs a chirality check to determine the best projection matrix (P2) that places the majority of
    the triangulated 3D points in front of both cameras.

    Args:
        P2s (list): List of candidate projection matrices for the second camera.
        x1 (np.ndarray): Homogeneous coordinates of keypoints in the first image (3xN).
        x2 (np.ndarray): Homogeneous coordinates of keypoints in the second image (3xN).

    Returns:
        tuple:
            - best_P (np.ndarray): The best projection matrix for the second camera that satisfies the chirality constraint.
            - num_points_infront_of_cam (int): The number of triangulated 3D points in front of both cameras.
    """

    # Set P1 as the default projection matrix for the first camera
    P1 = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)

    # Triangulate 3D points for each candidate P2
    triangulated_points = []
    for _, P2 in enumerate(P2s):
        DLT_points = triangulate_3D_point_DLT([P1, P2], [x1, x2])
        triangulated_points.append(DLT_points)

    # Select the best projection matrix based on the number of points in front of both cameras
    best_idx, num_points_infront_of_cam = select_best_solution(
        triangulated_points, P1, P2s
    )
    best_P = P2s[best_idx]

    return best_P, num_points_infront_of_cam


def count_points_in_front_of_cam(P1: np.array, P2: np.array, points_3d: np.array):
    """
    Counts the number of 3D points that are in front of both cameras, as indicated by their z-coordinate being positive
    in both camera coordinate systems.

    Args:
        P1 (np.ndarray): Projection matrix for the first camera (3x4).
        P2 (np.ndarray): Projection matrix for the second camera (3x4).
        points_3d (np.ndarray): Triangulated 3D points in homogeneous coordinates (4xN).

    Returns:
        int: Number of points that are in front of both cameras.
    """

    count = 0
    # Iterate over all 3D points and check if they are in front of both cameras
    for X in points_3d.T:
        if (P2 @ X)[2] > 0 and (P1 @ X)[
            2
        ] > 0:  # Check the z-coordinate in both camera frames
            count += 1
    return count


def select_best_solution(triangulated_points: list, P1: np.array, P2s: list):
    """
    Selects the best projection matrix (P2) by determining which set of triangulated points
    results in the maximum number of points in front of both cameras.

    Args:
        triangulated_points (list): List of 3D points triangulated from different projection matrices.
        P1 (np.ndarray): Projection matrix for the first camera (3x4).
        P2s (list): List of candidate projection matrices for the second camera (3x4).

    Returns:
        tuple:
            - best_idx (int): Index of the best projection matrix.
            - max_point_in_front (int): The maximum number of points in front of both cameras.
    """

    max_point_in_front = -1
    best_idx = -1

    # Iterate over each set of triangulated points and count the number of points in front of both cameras
    for i, points_3d in enumerate(triangulated_points):
        count = count_points_in_front_of_cam(P1, P2s[i], points_3d)

        # Update the best projection matrix if more points are in front of both cameras
        if count > max_point_in_front:
            max_point_in_front = count
            best_idx = i

    return best_idx, max_point_in_front
