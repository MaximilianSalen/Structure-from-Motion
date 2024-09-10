import os
import time
import logging
import pickle
import numpy as np
import cv2


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


def normalize_K(K, xs):
    return np.linalg.inv(K) @ xs


def cartesian_to_homogeneous(cartesian_points):
    # Add a row of ones at the bottom of the cartesian_points matrix
    homogeneous_points = np.vstack(
        (cartesian_points, np.ones((1, cartesian_points.shape[1])))
    )
    return homogeneous_points


def homogeneous_to_cartesian(points: np.ndarray) -> np.ndarray:
    return points[:-1, :] / points[-1, :]


def skew_symmetric_mat(v: np.array) -> np.array:
    """Generates a skew-symmetric matrix from a 3D vector."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def find_correspondences(img_path: str, desc_X: np.array, X0: np.array, K: list):
    """
    Finds 2D-3D correspondences between image keypoints and reconstructed 3D points.

    Args:
        img_path (str): Path to the input image.
        desc_X (np.ndarray): Descriptors of matched 3D points.
        X0 (np.ndarray): Reconstructed 3D points.
        K (list): Camera intrinsic matrix.

    Returns:
        tuple: Corresponding 3D points and normalized 2D points in the image.
    """
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors, desc_X.T, k=2)

    # Filter good matches using the ratio test
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    # Extract corresponding 2D and 3D points
    x = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).T
    X = np.float32([X0[:, m.trainIdx] for m in good_matches]).T
    x_norm = normalize_K(K, cartesian_to_homogeneous(x))

    return X, x_norm


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
