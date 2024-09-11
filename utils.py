import os
import logging
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from src import (
    pflat,
    homogeneous_to_cartesian,
    triangulate_3D_point_DLT,
    filter_3D_points,
)


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


def project_points(Ps, X, imgs, xs):
    plt.figure(figsize=(20, 20))
    for i in range(2):
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


def visualize_sfm_results(
    K, absolute_rotations, refined_Ts, x_pairs, nr_images, colors
):
    """
    Visualizes the 3D points and camera positions in a 3D plot.

    Args:
        K (np.ndarray): Camera intrinsic matrix.
        absolute_rotations (list): List of absolute rotation matrices for each camera.
        refined_Ts (list): List of refined translation vectors for each camera.
        x_pairs (list): List of keypoint pairs between consecutive images.
        nr_images (int): Number of images in the sequence.
        colors (list): List of colors to use for the camera visualizations.
    """
    # Set up the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Points and Camera Positions")
    ax.axis("equal")

    # Loop through the camera positions and visualize the points and cameras
    for i_camera in range(nr_images - 1):
        P1 = np.hstack(
            (absolute_rotations[i_camera], refined_Ts[i_camera].reshape(3, 1))
        )
        P2 = np.hstack(
            (absolute_rotations[i_camera + 1], refined_Ts[i_camera + 1].reshape(3, 1))
        )

        P1 = K @ P1
        P2 = K @ P2

        x1 = x_pairs[2 * i_camera]
        x2 = x_pairs[2 * i_camera + 1]

        # Triangulate 3D points and filter them
        X_triangulated = triangulate_3D_point_DLT([P1, P2], [x1, x2])
        X_filtered = filter_3D_points(X_triangulated)

        # Plot the filtered 3D points and camera positions
        plot_3d_points_and_cameras_new(X_filtered, [P1, P2], ax, colors[i_camera])

    # Display the 3D plot
    plt.show()
