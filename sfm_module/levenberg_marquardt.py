import numpy as np


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


# def levenberg_marquardt_optimize_T(K, R, X, x_norm, T_initial, num_iterations, mu):
#     T = np.copy(T_initial)

#     for _ in range(num_iterations):
#         error_tot = np.array([])
#         J_tot = np.array([]).reshape(0, 3)

#         for j in range(X.shape[1]):
#             # Get error
#             error_j = compute_reprojection_error(X[:, j], x_norm[:, j], K, R, T)
#             # Get jacobian
#             J_j = projection_derivatives_wrt_T(X[:, j], K, R, T)

#             # stack error and J for the j_th 3D point
#             error_tot = np.concatenate((error_tot, error_j))
#             J_tot = np.vstack((J_tot, J_j))

#         delta_T = ComputeUpdate(error_tot, J_tot, mu)
#         T += delta_T

#     return np.reshape(T, (3,1))

# def compute_reprojection_error(X, x_norm, K, R, T):
#     x_projected = R @ X + T

#     x_projected /= x_projected[2] # Normalize to 2D

#     error = x_projected[:2] - x_norm[:2]
#     return error


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


def cartesian_to_homogeneous(cartesian_points):
    # Add a row of ones at the bottom of the cartesian_points matrix
    homogeneous_points = np.vstack(
        (cartesian_points, np.ones((1, cartesian_points.shape[1])))
    )
    return homogeneous_points
