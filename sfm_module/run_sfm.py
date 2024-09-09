import argparse
import matplotlib.pyplot as plt
import time
import logging
import auxiliary
import numpy as np
import yaml
import os
from tqdm import tqdm
from ransac_algorithm import run_ransac
from extract_sift import extract_sift_data
from reconstruct_3D import run_reconstruction
from estimate_T import get_T, get_correspondences
from levenberg_marquardt import levenberg_marquardt_optimize_T

# Set up logging
logging.basicConfig(level=logging.INFO)


def get_data(path_to_cfg: str):
    cfg_path = os.path.join(path_to_cfg, "cfg.yml")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r") as file:
            cfg_file = yaml.safe_load(file)
            focal_length = cfg_file["camera"]["focal_length"]
            principal_point = cfg_file["camera"]["principal_point"]
            img_paths = [path for path in cfg_file["image_file_paths"]]
            init_pair = cfg_file["initial_pair"]
        K = [
            [focal_length[0], 0, principal_point[0]],
            [0, focal_length[1], principal_point[1]],
            [0, 0, 1],
        ]
    else:
        raise OSError("File not found")

    return K, img_paths, init_pair


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

    # Call the extract_sift_data function
    x_pairs, initial_pair = extract_sift_data(
        img_names=img_names,
        init_pair=init_pair,
        dataset=args.dataset,
    )

    # i_R, i_T, inss = run_ransac(K, initial_pair[0], initial_pair[1], pixel_threshold)

    RT_list = []
    R_list = []
    T_list = []
    for i in range(len(x_pairs) // 2):
        start_time = time.time()
        x1 = x_pairs[2 * i]
        x2 = x_pairs[2 * i + 1]
        i_R, i_T, _ = run_ransac(K, x1, x2, pixel_threshold)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed Time Run RANSAC: {elapsed_time} seconds")
        i_RT = [i_R, i_T]
        RT_list = RT_list + i_RT
        R_list = R_list + [i_R]
        T_list = T_list + [i_T]

    # # Plot Testing code
    # for i_camera in range(nr_images-1):
    #     P1 = np.concatenate((np.eye(3), np.zeros((3,1))), axis=1)
    #     P2 = np.hstack((R_list[i_camera], T_list[i_camera]))
    #     P1 = K @ P11

    #     P2 = K @ P2
    #     x1 = x_pairs[i_camera]
    #     x2 = x_pairs[i_camera+1]
    #     X_triangulated = auxiliary.triangulate_3D_point_DLT([P1, P2], [x1, x2])
    #     auxiliary.plot_3d_points_and_cameras(X_triangulated, [P1, P2])

    # im1 = cv2.imread(img_names[0])
    # im2 = cv2.imread(img_names[1])
    # auxiliary.project_points([P1, P2], X_triangulated, [im1, im2], [x1, x2])

    X0, absolute_rotations, inliers = run_reconstruction(
        R_list, initial_pair[0], initial_pair[1], K, init_pair[0], pixel_threshold
    )
    desc_X = initial_pair[2]
    desc_X_inliers = desc_X[:, inliers]
    print("Sum of inliers :", np.sum(inliers))

    if np.sum(inliers) < 50:
        print("Insufficient amount of inliers change initial pair")
        exit

    # estimate T robustly and store3
    initial_Ts = []
    start_time = time.time()
    for i in range(nr_images):
        # get 2d_3d correspondences
        X_corr, x_corr_norm = get_correspondences(img_names[i], desc_X, X0, K)
        init_T = get_T(x_corr_norm, X_corr, K, absolute_rotations[i], pixel_threshold)
        initial_Ts = initial_Ts + [init_T]
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time Estimate T: {elapsed_time} seconds")

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
            img_names[i], desc_X_inliers, X0, K
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
        X_triangulated = auxiliary.triangulate_3D_point_DLT([P1, P2], [x1, x2])
        X_filtered = auxiliary.filter_3D_points(X_triangulated)
        auxiliary.plot_3d_points_and_cameras(X_filtered, [P1, P2])

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
        X_triangulated = auxiliary.triangulate_3D_point_DLT([P1, P2], [x1, x2])
        X_filtered = auxiliary.filter_3D_points(X_triangulated)
        auxiliary.plot_3d_points_and_cameras_new(
            X_filtered, [P1, P2], ax, colors[i_camera]
        )
    plt.show()

    print("Test completed")


run_sfm()
