import numpy as np
from utils import *


def optimize_translation(
    K,
    img_paths,
    desc_X_inliers,
    X0,
    estimated_Ts,
    absolute_rotations,
):
    X_corr_inliers = []
    x_corr_norm_inliers = []
    for i in range(len(img_paths)):

        # Find 2D-3D correspondences with only inliers
        X_corr_inlier, x_corr_norm_inlier = find_correspondences(
            img_paths[i], desc_X_inliers, X0, K
        )
        X_corr_inliers.append(X_corr_inlier)
        x_corr_norm_inliers.append(x_corr_norm_inlier)

    # Refine T with Levenberg_marquardt
    refined_Ts = []
    for i, estimated_T in enumerate(estimated_Ts):
        refined_T = levenberg_marquardt_optimize_T(
            K,
            absolute_rotations[i],
            X_corr_inliers[i],
            x_corr_norm_inliers[i],
            estimated_T,
            num_iterations=10,
            mu=0.01,
        )
        refined_Ts = refined_Ts + [refined_T]


def levenberg_marquardt_optimize_T(K, R, X, x_norm, T_initial, num_iterations, mu):
    T = np.copy(T_initial)

    for _ in range(num_iterations):
        current_error = compute_reprojection_error(X, x_norm, K, R, T)

        # Get the jacobian
        J_tot = np.array([]).reshape(0, 3)
        for j in range(X.shape[1]):
            # Get jacobian
            J_j = projection_derivatives_wrt_T(X[:, j], K, R, T)
            J_tot = np.vstack((J_tot, J_j))

        # Update delta T with current error
        delta_T = ComputeUpdate(current_error, J_tot, mu)
        new_T = T + delta_T

        # calculate new error and compare with current error. If new error is lower than make T to new_T
        new_error = compute_reprojection_error(X, x_norm, K, R, new_T)
        if np.sum(new_error**2) < np.sum(current_error**2):
            T = new_T

    return np.reshape(T, (3, 1))


def compute_reprojection_error(X, x_norm, K, R, T):
    x_projected = R @ X + T[:, np.newaxis]

    x_projected /= x_projected[2]  # Normalize to 2D

    errors = x_projected[:2, :] - x_norm[:2, :]
    return errors.flatten()


def ComputeUpdate(error, J, mu):
    C = J.T @ J + mu * np.eye(J.shape[1])
    c = J.T @ error
    return np.linalg.solve(-C, c)


def projection_derivatives_wrt_T(X, K, R, T):
    X_cam = R @ X + T
    x_proj = K @ X_cam

    # Initialize Jacobian
    J = np.zeros((2, 3))

    # Iterate over T_i (T_x, T_y, T_z)
    for i in range(3):
        dX_cam_dTi = np.zeros(3)
        dX_cam_dTi[i] = 1

        # Compute the derivative of the projected point in homogenous coords
        dx_proj_dTi = K @ dX_cam_dTi

        # Chain rule (by using the quotient rule)
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
