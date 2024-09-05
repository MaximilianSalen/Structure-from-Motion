import matplotlib.pyplot as plt
import time
import auxiliary
import numpy as np
from get_dataset_info import get_dataset_info
from ransac_algorithm import run_ransac
from extract_sift import execute_sift_extraction
from reconstruct_3D import run_reconstruction
from estimate_T import get_T, get_correspondences
from levenberg_marquardt import levenberg_marquardt_optimize_T


def test():

    print("Enter a dataset (1-11):")
    dataset_nr = int(input())
    data_root = "/home/simsom/work_space/repos/Structure-from-motion/sfm_module/"
    K, img_names, init_pair, pixel_threshold = get_dataset_info(data_root, dataset_nr)
    nr_images = len(img_names)

    # File name for saving/loading x_pairs
    x_pairs_filename = f"x_pairs_dataset_{dataset_nr}.pkl"

    # Check if x_pairs already exist
    x_pairs = auxiliary.load_x_pairs(x_pairs_filename)

    if x_pairs is None:
        x_pairs = []
        for i in range(nr_images - 1):
            start_time = time.time()
            x1, x2, _ = execute_sift_extraction(img_names[i], img_names[i + 1])
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed Time Sift extraction: {elapsed_time} seconds")
            x_pairs = x_pairs + [x1, x2]

        # Save the extracted x_pairs
        auxiliary.save_x_pairs(x_pairs, x_pairs_filename)
    else:
        print("x_pairs data already exists. Loaded from file.")

    ### Initial Pair ###
    init_pair_filename = f"init_pair_dataset_{dataset_nr}.pkl"
    initial_pair = auxiliary.load_x_pairs(init_pair_filename)

    if initial_pair is None:
        start_time = time.time()

        init_imgs = [img_names[i] for i in init_pair]
        init_x1, init_x2, desc_X = execute_sift_extraction(init_imgs[0], init_imgs[1])

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed Time Sift extraction: {elapsed_time} seconds")
        initial_pair = [init_x1, init_x2, desc_X]

        # Save the extracted initial pair
        auxiliary.save_x_pairs(initial_pair, init_pair_filename)
    else:
        print("initial_pairs data already exists. Loaded from file.")

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


test()
