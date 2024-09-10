import os
import pickle
import time
import logging
import numpy as np
import cv2
import yaml


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


def get_data(path_to_cfg: str):
    """
    Loads camera parameters and image data from a configuration file.

    Args:
        path_to_cfg (str): Directory path to the 'cfg.yml' file.

    Returns:
        Dict containing:
            - K (list): 3x3 intrinsic camera matrix.
            - img_names (list): List of image file names.
            - init_pair (list): Indices for initial image pair from config.

    Raises:
        OSError: If 'cfg.yml' is not found.
    """

    cfg_path = os.path.join(path_to_cfg, "cfg.yml")

    if not os.path.isfile(cfg_path):
        raise OSError("File not found")

    with open(cfg_path, "r") as file:
        cfg_file = yaml.safe_load(file)

        focal_length = cfg_file["camera"]["focal_length"]
        principal_point = cfg_file["camera"]["principal_point"]
        img_names = cfg_file["image_file_names"]
        init_pair = cfg_file["initial_pair"]

    # Constructing the intrinsic camera matrix
    K = [
        [focal_length[0], 0, principal_point[0]],
        [0, focal_length[1], principal_point[1]],
        [0, 0, 1],
    ]

    # Create paths to images
    img_paths = [os.path.join(path_to_cfg, img_name) for img_name in img_names]

    return K, img_paths, init_pair


def setup_logging(verbosity=None):
    """
    Configures the logging settings based on the verbosity level.
    If verbosity is None, logging is disabled.

    Args:
        verbosity (str): Logging level as a string (DEBUG, INFO, etc.), or None to disable logging.
    """
    if verbosity:
        logging.basicConfig(
            level=getattr(logging, verbosity),
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()],
        )
    else:
        # Disable logging by setting the logging level to CRITICAL (no lower levels will be shown)
        logging.disable(logging.CRITICAL)
