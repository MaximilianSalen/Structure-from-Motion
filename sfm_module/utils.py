import os
import pickle
import time
import logging
import cv2
import numpy as np
import scipy
from scipy.linalg import null_space
import matplotlib.pyplot as plt


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


def plot_3d_points_and_cameras_new(X, P, ax, color):
    """
    Plots the 3D points and cameras on the given axis.
    X: 4xN matrix of 3D points
    P: List of camera matrices
    ax: Matplotlib axis object for plotting
    """
    # Convert homogeneous coordinates to 3D coordinates
    X_3d = pflat(X)

    # Plotting the 3D points
    ax.scatter(X_3d[0, :], X_3d[1, :], X_3d[2, :], s=1, color=color)

    # Plotting the cameras
    num_cams = len(P)
    c = np.zeros((4, num_cams))
    v = np.zeros((3, num_cams))

    for i in range(num_cams):
        ns = null_space(P[i])
        if ns.size:  # Check if null space is not empty
            c[:, i] = ns[:, 0]
        v[:, i] = P[i][2, 0:3]

    c = c / c[3, :]
    ax.quiver(
        c[0, :], c[1, :], c[2, :], v[0, :], v[1, :], v[2, :], color="r", linewidth=1.5
    )


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


def plot_3d_points_and_cameras(X, P):
    """
    Plots the 3D points and cameras.
    X: 4xN matrix of 3D points
    P: List of camera matrices
    """
    # Convert homogeneous coordinates to 3D coordinates
    X_3d = pflat(X)

    # Plotting the 3D points
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X_3d[0, :], X_3d[1, :], X_3d[2, :], s=1)

    # # Plotting the cameras
    num_cams = len(P)
    c = np.zeros((4, num_cams))
    v = np.zeros((3, num_cams))

    for i in range(num_cams):
        ns = null_space(P[i])
        if ns.size:  # Check if null space is not empty
            c[:, i] = ns[:, 0]
        v[:, i] = P[i][2, 0:3]

    c = c / c[3, :]
    ax.quiver(
        c[0, :], c[1, :], c[2, :], v[0, :], v[1, :], v[2, :], color="r", linewidth=1.5
    )

    # Setting axis properties for a realistic view
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Points and Camera Positions")
    ax.axis("equal")

    plt.show()


def pflat(X):
    """Normalize a matrix by dividing each column by its last element."""
    return X / X[-1, :]


def cartesian_to_homogeneous(cartesian_points):
    # Add a row of ones at the bottom of the cartesian_points matrix
    homogeneous_points = np.vstack(
        (cartesian_points, np.ones((1, cartesian_points.shape[1])))
    )
    return homogeneous_points


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


def draw_points(image_name, points):
    # Load the image
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw each point on the image
    for point in points.T:
        x, y = point[:2]  # Assuming each point is a tuple (x, y)
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Display the image
    plt.imshow(image)
    plt.show()


def project_points(Ps, X, imgs, xs):
    plt.figure(figsize=(20, 20))
    for i in range(2):
        # X_homo = cartesian_to_homogeneous(X)
        projected_points = Ps[i] @ X
        X_projected = homogeneous_to_cartesian(projected_points)

        x = xs[i]
        plt.subplot(1, 2, i + 1)  # Create a subplot with 1 row and 2 columns
        plt.imshow(imgs[i])
        plt.scatter(
            X_projected[0, :],
            X_projected[1, :],
            s=14,
            color="magenta",
            label="Projected",
        )
        plt.scatter(
            x[0, :],
            x[1, :],
            s=10,
            marker="x",
            color="blue",
            label="SIFT",
            linewidth=0.7,
        )
        plt.legend()
        plt.title(f"Cube {i + 1}")
        plt.xlim([0, imgs[i].shape[1]])
        plt.ylim([imgs[i].shape[0], 0])
    plt.show()


def homogeneous_to_cartesian(points: np.ndarray) -> np.ndarray:
    return points[:-1, :] / points[-1, :]


def load_data(file_path_data: str):
    data = scipy.io.loadmat(file_path_data, squeeze_me=True)
    return data
