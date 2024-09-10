import os
import argparse
import time
import logging
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import *
from visualization import *
from ransac_algorithm import estimate_R
from extract_sift import process_sift_for_image_pairs
from reconstruct_3D import run_reconstruction
from estimate_translation import estimate_translation
from levenberg_marquardt import optimize_translation


def run_sfm():
    """Main function to run the structure from motion pipeline."""

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run Structure-from-Motion pipeline.")
    parser.add_argument("data_path", type=str, help="Base path to the dataset")
    parser.add_argument("dataset", type=str, help="Name of the dataset")
    parser.add_argument("threshold", type=float, help="Pixel threshold for matching")
    parser.add_argument(
        "--verbosity",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: CRITICAL)",
    )
    args = parser.parse_args()
    dataset_path = os.path.join(args.data_path, args.dataset)

    setup_logging(args.verbosity)
    logging.info(f"Dataset path: {dataset_path}")

    # Load necessary data
    K, img_paths, init_pair = get_data(dataset_path)
    pixel_threshold = args.threshold
    nr_images = len(img_paths)

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
    estimated_Ts = estimate_translation(
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

    refined_Ts = optimize_translation()
    for i_camera in range(nr_images - 1):
        P1 = np.hstack(
            (absolute_rotations[i_camera], np.reshape(estimated_Ts[i_camera], (3, 1)))
        )
        P2 = np.hstack(
            (
                absolute_rotations[i_camera + 1],
                np.reshape(estimated_Ts[i_camera + 1], (3, 1)),
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
