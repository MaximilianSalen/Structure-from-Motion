import os
import logging
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import null_space
from src import (
    pflat,
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


def plot_3d_points_and_cameras(X, P, ax, color):
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


def visualize_sfm_results_with_rotation(
    sfm_data_dict,
    dataset_name,
    colors,
):
    """
    Visualizes the 3D points and camera positions in a rotating 3D plot, and saves it as an animated GIF.

    Args:
        sfm_data_dict (dict): Dictionary containing necessary information for visualization of the 3D construction.
        colors (list): List of colors to use for the camera visualizations.
    """
    K = sfm_data_dict["K"]
    absolute_rotations = sfm_data_dict["absolute_rotations"]
    refined_Ts = sfm_data_dict["refined_Ts"]
    x_pairs = sfm_data_dict["x_pairs"]
    nr_images = sfm_data_dict["nr_images"]

    def rotate(angle):
        ax.view_init(elev=30, azim=angle, vertical_axis="y")

    # Create a figure and a 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Points and Camera Positions")
    ax.invert_yaxis()  # Invert Y-axis
    # ax.axis("equal")

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
        plot_3d_points_and_cameras(X_filtered, [P1, P2], ax, colors[i_camera])

    # Create a progress bar for the rotation
    pbar = tqdm(total=120, desc="Generating Frames", unit="frame")

    def update_frame(angle):
        rotate(angle)
        pbar.update(1)  # Update the progress bar

    # Create the animation
    rot_animation = animation.FuncAnimation(
        fig, update_frame, frames=np.arange(0, 360, 3), interval=100
    )

    if not os.path.exists("output"):
        os.makedirs("output", exist_ok=True)
    rot_animation.save(
        f"output/rotation_dataset_{dataset_name}.gif", dpi=60, writer="pillow"
    )
    pbar.close()
    print(f"Animation saved to output/rotation_dataset_{dataset_name}.gif")
