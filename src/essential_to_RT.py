import numpy as np
import scipy
from .chirality_check import perform_chirality_check


def essential_to_RT(E, K, x1, x2):
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    U, S, Vt = scipy.linalg.svd(E)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Vt = -Vt
    P1 = np.hstack((U @ W @ Vt, U[:, -1].reshape(-1, 1)))
    P2 = np.hstack((U @ W @ Vt, -U[:, -1].reshape(-1, 1)))
    P3 = np.hstack((U @ W.T @ Vt, U[:, -1].reshape(-1, 1)))
    P4 = np.hstack((U @ W.T @ Vt, -U[:, -1].reshape(-1, 1)))

    best_P, num_points_infront_of_cam = perform_chirality_check(
        [P1, P2, P3, P4], K, x1, x2
    )
    best_R = best_P[:, :3]
    best_T = best_P[:, 3]
    return best_R, best_T, num_points_infront_of_cam
