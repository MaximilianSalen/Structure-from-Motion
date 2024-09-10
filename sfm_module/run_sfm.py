import argparse
import matplotlib.pyplot as plt
import time
import logging
from utils import *
from visualization import *
import numpy as np
import yaml
import os
from tqdm import tqdm
from ransac_algorithm import estimate_R
from extract_sift import process_sift_for_image_pairs
from reconstruct_3D import run_reconstruction
from estimate_translation import estimate_translation
from levenberg_marquardt import levenberg_marquardt_optimize_T

# Set up logging
logging.basicConfig(level=logging.INFO)


def get_data(path_to_cfg: str):
    """
    Loads camera parameters and image data from a configuration file.

    Args:
        path_to_cfg (str): Directory path to the 'cfg.yml' file.

    Returns:
        tuple: A tuple containing:
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

    return K, img_names, init_pair


def run_sfm():
    """Main function to run the structure from motion pipeline."""

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run Structure-from-Motion pipeline.")
    parser.add_argument("data_path", type=str, help="Base path to the dataset")
    parser.add_argument("dataset", type=str, help="Name of the dataset")
    parser.add_argument("threshold", type=float, help="Pixel threshold for matching")
    args = parser.parse_args()
    dataset_path = os.path.join(args.data_path, args.dataset)
    logging.info(f"Dataset path: {dataset_path}")

    # Load necessary data
    K, img_names, init_pair = get_data(dataset_path)
    pixel_threshold = args.threshold
    nr_images = len(img_names)
    img_paths = [os.path.join(dataset_path, img_name) for img_name in img_names]

    # Call the extract_sift_data function
    x_pairs, init_pair_dict = process_sift_for_image_pairs(
        img_paths=img_paths,
        init_pair=init_pair,
        dataset=args.dataset,
    )

    # Run RANSAC algorithm to estimate relative rotations between consecutive image pairs
    logging.info("Running RANSAC to estimate relative rotations between image pairs...")
    R_list = estimate_R(K, x_pairs, pixel_threshold)

    # Reconstruct the 3D model from initial pair
    logging.info("Reconstructing the 3D model from the initial pair...")
    X0, absolute_rotations, inliers = run_reconstruction(
        R_list, init_pair_dict, K, pixel_threshold
    )
    desc_X = init_pair_dict["init_pair_desc"]
    desc_X_inliers = desc_X[:, inliers]

    num_inliers = np.sum(inliers)
    logging.info(f"Total number of inliers for initial reconstruction: {num_inliers}")

    if num_inliers < 50:
        logging.error(
            "Insufficient number of inliers (<50). Possible solution: change initial pair. Exiting..."
        )
        exit()
    else:
        logging.info("Sufficient inliers found. Proceeding with reconstruction.")

    # Robustly estimate T
    initial_Ts = estimate_translation(
        K, desc_X, X0, absolute_rotations, img_paths, pixel_threshold
    )

    ############
    # for i_camera in range(nr_images-1):
    #     P1 = np.hstack((absolute_rotations[i_camera], np.reshape(initial_Ts[i_camera], (3,1))))
    #     P2 = np.hstack((absolute_rotations[i_camera+1], np.reshape(initial_Ts[i_camera+1], (3,1))))
    #     P1 = K @ P1
    #     P2 = K @ P2
    #     x1 = x_pairs[2*i_camera]
    #     x2 = x_pairs[2*i_camera+1]
    #     X_triangulated = auxiliary.triangulate_3D_point_DLT([P1, P2], [x1, x2])
    #     X_filtered = auxiliary.filter_3D_points(X_triangulated)
    #     auxiliary.plot_3d_points_and_cameras(X_filtered, [P1, P2])

    X_corr_inliers = []
    x_corr_norm_inliers = []
    for i in range(nr_images):
        # get 2d_3d correspondences with only inliers
        X_corr_inlier, x_corr_norm_inlier = get_correspondences(
            img_paths[i], desc_X_inliers, X0, K
        )
        X_corr_inliers.append(X_corr_inlier)
        x_corr_norm_inliers.append(x_corr_norm_inlier)

    # Refine T with Levenberg_marquardt
    refined_Ts = []
    start_time = time.time()
    for i, init_T in enumerate(initial_Ts):
        refined_T = levenberg_marquardt_optimize_T(
            K,
            absolute_rotations[i],
            X_corr_inliers[i],
            x_corr_norm_inliers[i],
            init_T,
            num_iterations=10,
            mu=0.01,
        )
        refined_Ts = refined_Ts + [refined_T]
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time Refine T: {elapsed_time} seconds")

    for i_camera in range(nr_images - 1):
        P1 = np.hstack(
            (absolute_rotations[i_camera], np.reshape(initial_Ts[i_camera], (3, 1)))
        )
        P2 = np.hstack(
            (
                absolute_rotations[i_camera + 1],
                np.reshape(initial_Ts[i_camera + 1], (3, 1)),
            )
        )
        P1 = K @ P1
        P2 = K @ P2
        x1 = x_pairs[2 * i_camera]
        x2 = x_pairs[2 * i_camera + 1]
        X_triangulated = triangulate_3D_point_DLT([P1, P2], [x1, x2])
        X_filtered = filter_3D_points(X_triangulated)
        plot_3d_points_and_cameras(X_filtered, [P1, P2])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Points and Camera Positions")
    ax.axis("equal")

    colors = [
        "blue",
        "green",
        "red",
        "cyan",
        "magenta",
        "yellow",
        "black",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "navy",
        "lightblue",
        "lightgreen",
        "coral",
        "beige",
        "indigo",
        "lime",
        "maroon",
        "teal",
        "rose",
        "mustard",
        "turquoise",
        "sienna",
        "plum",
        "orchid",
        "lavender",
    ]
    for i_camera in range(nr_images - 1):
        P1 = np.hstack((absolute_rotations[i_camera], refined_Ts[i_camera]))
        P2 = np.hstack((absolute_rotations[i_camera + 1], refined_Ts[i_camera + 1]))
        P1 = K @ P1
        P2 = K @ P2
        x1 = x_pairs[2 * i_camera]
        x2 = x_pairs[2 * i_camera + 1]
        X_triangulated = triangulate_3D_point_DLT([P1, P2], [x1, x2])
        X_filtered = filter_3D_points(X_triangulated)
        plot_3d_points_and_cameras_new(X_filtered, [P1, P2], ax, colors[i_camera])
    plt.show()

    print("Test completed")


if __name__ == "__main__":
    run_sfm()
