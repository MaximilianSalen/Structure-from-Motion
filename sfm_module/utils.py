import os
import pickle
import time
import logging
import numpy as np


def log_execution_time(func):
    """
    A decorator to log the execution time of a function.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logging.info(f"Elapsed Time for {func.__name__}: {elapsed_time:.2f} seconds")
        return result

    return wrapper


def filter_3D_points(X):
    X_mean = np.mean(X, axis=0)
    distances = np.linalg.norm(X - X_mean, axis=1)
    quantile_90 = np.quantile(distances, 0.9)
    filtered_X = X[distances <= 5 * quantile_90]
    return filtered_X


def triangulate_3D_point_DLT(P, points):
    triangulated_points = []
    points1, points2 = points
    P1, P2 = P
    # Construct the system of equations
    for i in range(points1.shape[1]):
        A = np.zeros((4, 4))
        A[0] = points1[0][i] * P1[2] - P1[0]
        A[1] = points1[1][i] * P1[2] - P1[1]
        A[2] = points2[0][i] * P2[2] - P2[0]
        A[3] = points2[1][i] * P2[2] - P2[1]

        # Solve for X: AX = 0
        U, S, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]  # Convert from homogeneous to 3D coordinates but still keep 4 dim

        triangulated_points.append(X)

    return np.array(triangulated_points).T


def pflat(X):
    """Normalize a matrix by dividing each column by its last element."""
    return X / X[-1, :]


def cartesian_to_homogeneous(cartesian_points):
    # Add a row of ones at the bottom of the cartesian_points matrix
    homogeneous_points = np.vstack(
        (cartesian_points, np.ones((1, cartesian_points.shape[1])))
    )
    return homogeneous_points


def homogeneous_to_cartesian(points: np.ndarray) -> np.ndarray:
    return points[:-1, :] / points[-1, :]


# Function to save x_pairs using pickle
def save_x_pairs(data, filename, save_location):
    file_path = os.path.join(save_location, filename)
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


# Function to load x_pairs if file exists
def load_x_pairs(filename, save_location):
    file_path = os.path.join(save_location, filename)
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            return pickle.load(file)
    return None