import numpy as np
from .utils import *


def refine_translation(
    K: list,
    img_paths: list,
    desc_X_inliers: np.array,
    X0: np.array,
    absolute_rotations: list,
    estimated_Ts: list,
):
    """
    Refines the translation vectors using inlier 2D-3D correspondences and the Levenberg-Marquardt algorithm.

    Args:
        K (list): Camera intrinsic matrix.
        img_paths (list): List of paths to the images.
        desc_X_inliers (np.ndarray): Descriptors of the inlier 3D points.
        X0 (np.ndarray): Reconstructed 3D points.
        absolute_rotations (list): List of absolute rotation matrices for each camera.
        estimated_Ts (list): Initial estimates for the translation vectors.

    Returns:
        list: Refined translation vectors (refined_Ts).
    """

    X_corr_inliers = []
    x_corr_norm_inliers = []
    for i in range(len(img_paths)):
        # Find 2D-3D correspondences using only inliers
        X_corr_inlier, x_corr_norm_inlier = find_correspondences(
            img_paths[i], desc_X_inliers, X0, K
        )
        X_corr_inliers.append(X_corr_inlier)
        x_corr_norm_inliers.append(x_corr_norm_inlier)

    # Refine the translation vectors using the Levenberg-Marquardt algorithm
    refined_Ts = []
    for i, estimated_T in enumerate(estimated_Ts):
        refined_T = levenberg_marquardt_algorithm(
            K,
            absolute_rotations[i],
            X_corr_inliers[i],
            x_corr_norm_inliers[i],
            estimated_T,
            num_iterations=10,
            mu=0.01,
        )
        refined_Ts.append(refined_T)


def levenberg_marquardt_algorithm(
    K: list,
    R: np.array,
    X: np.array,
    x_norm: np.array,
    T_initial: np.array,
    num_iterations: int,
    mu: float,
) -> np.array:
    """
    Refines the translation vector using the Levenberg-Marquardt algorithm.

    Args:
        K (list): Camera intrinsic matrix.
        R (np.ndarray): Rotation matrix for the current camera.
        X (np.ndarray): 3D points corresponding to the inliers.
        x_norm (np.ndarray): Normalized 2D points.
        T_initial (np.ndarray): Initial translation vector.
        num_iterations (int): Number of iterations for the optimization.
        mu (float): Damping parameter for the Levenberg-Marquardt algorithm.

    Returns:
        np.ndarray: Refined translation vector.
    """

    T = np.copy(T_initial)

    for _ in range(num_iterations):
        # Compute the current reprojection error
        current_error = compute_reprojection_error(X, x_norm, K, R, T)

        # Compute the Jacobian matrix
        J_tot = np.array([]).reshape(0, 3)
        for j in range(X.shape[1]):
            # Derivatives of the projection with respect to T
            J_j = projection_derivatives_wrt_T(X[:, j], K, R, T)
            J_tot = np.vstack((J_tot, J_j))

        # Update T using the current error and Jacobian
        delta_T = ComputeUpdate(current_error, J_tot, mu)
        new_T = T + delta_T

        # Calculate new error and update T if the new error is lower
        new_error = compute_reprojection_error(X, x_norm, K, R, new_T)
        if np.sum(new_error**2) < np.sum(current_error**2):
            T = new_T

    return np.reshape(T, (3, 1))


def compute_reprojection_error(
    X: np.array, x_norm: np.array, K: list, R: np.array, T: np.array
) -> np.array:
    """
    Computes the reprojection error between 3D points and 2D normalized points.

    Args:
        X (np.ndarray): 3D points.
        x_norm (np.ndarray): Normalized 2D points.
        K (list): Camera intrinsic matrix.
        R (np.ndarray): Rotation matrix for the current camera.
        T (np.ndarray): Translation vector for the current camera.

    Returns:
        np.ndarray: Flattened array of reprojection errors.
    """

    # Project 3D points using the current rotation and translation
    x_projected = R @ X + T[:, np.newaxis]

    # Normalize the projected points
    x_projected /= x_projected[2]

    # Compute the reprojection error
    errors = x_projected[:2, :] - x_norm[:2, :]
    return errors.flatten()


def ComputeUpdate(error: np.array, J: np.array, mu: float) -> np.array:
    """
    Computes the update for the translation vector using the Levenberg-Marquardt method.

    Args:
        error (np.ndarray): Current reprojection error.
        J (np.ndarray): Jacobian matrix.
        mu (float): Damping parameter for the Levenberg-Marquardt algorithm.

    Returns:
        np.ndarray: The update for the translation vector.
    """

    # Compute the Hessian approximation
    C = J.T @ J + mu * np.eye(J.shape[1])

    # Compute the gradient
    c = J.T @ error

    # Solve for the update
    return np.linalg.solve(-C, c)


def projection_derivatives_wrt_T(
    X: np.array, K: np.array, R: np.array, T: np.array
) -> np.array:
    """
    Computes the derivatives of the projection function with respect to the translation vector T.

    Args:
        X (np.ndarray): 3D point.
        K (np.ndarray): Camera intrinsic matrix.
        R (np.ndarray): Rotation matrix for the current camera.
        T (np.ndarray): Translation vector for the current camera.

    Returns:
        np.ndarray: Jacobian matrix representing the projection derivatives with respect to T.
    """

    # Project the 3D point using the current rotation and translation
    X_cam = R @ X + T
    x_proj = K @ X_cam

    # Initialize the Jacobian matrix
    J = np.zeros((2, 3))

    # Compute the Jacobian entries for each component of T
    for i in range(3):
        dX_cam_dTi = np.zeros(3)
        dX_cam_dTi[i] = 1

        # Derivative of the projected point in homogeneous coordinates
        dx_proj_dTi = K @ dX_cam_dTi

        # Apply the chain rule to compute the Jacobian
        dx_z_inv_squared = 1 / (x_proj[2] ** 2)
        dx1_norm_dTi = (
            x_proj[2] * dx_proj_dTi[0] - x_proj[0] * dx_proj_dTi[2]
        ) * dx_z_inv_squared
        dx2_norm_dTi = (
            x_proj[2] * dx_proj_dTi[1] - x_proj[1] * dx_proj_dTi[2]
        ) * dx_z_inv_squared
        J[0, i] = dx1_norm_dTi
        J[1, i] = dx2_norm_dTi

    return J
