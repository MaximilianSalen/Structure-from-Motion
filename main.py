"""
Computer Vision
EEN020
Project
2023-12-31

Executes the SFM pipeline

Authors:
        Maximilian Salén
        Axel Qvarnström
"""

import numpy as np
from get_dataset_info import get_dataset_info
from ransac_algorithm import run_ransac
from extract_sift import execute_sift_extraction
from reconstruct_3D import run_reconstruction
from estimate_T import get_T_and_correspondences
from auxiliary import triangulate_3D_point_DLT, plot_3d_points_and_cameras, cartesian_to_homogeneous
from levenberg_marquardt import levenberg_marquardt_optimize_T

def main():
    print('Enter a dataset (1-9):')
    dataset_nr = input()
    K, img_names, init_pair, pixel_threshold = get_dataset_info(dataset_nr)
    nr_images = len(img_names)

    RT_list = []
    x_pairs = []
    for i in range(nr_images-1):
        x1, x2 = execute_sift_extraction(img_names[i], img_names[i+1])
        i_R, i_T, _ = run_ransac(K, x1, x2, pixel_threshold)
        i_RT = [i_R, i_T]
        RT_list = RT_list + i_RT
        x_pairs = x_pairs + [x1, x2]

    init_imgs = [img_names[i] for i in init_pair]
    X0, desc_X, absolute_rotations = run_reconstruction(RT_list, init_imgs, K, init_pair[0])

    # estimate T robustly and store
    initial_Ts = []
    for i in range(nr_images):
        init_T, X_corrs, x_corrs = get_T_and_correspondences(img_names[i], desc_X, X0, K, absolute_rotations[i], pixel_threshold)
        initial_Ts = initial_Ts + init_T
        
    # Refine T with Levenberg_marquardt
    refined_Ts = []
    for i, init_T in enumerate(initial_Ts):
        refined_T = levenberg_marquardt_optimize_T(K, absolute_rotations[i], X_corrs, x_corrs, init_T, num_iterations=100, mu=0.01)
        refined_Ts = refined_Ts + refined_T
    
    
    for i_camera in range(nr_images):
        P = np.concatenate((absolute_rotations[i_camera], refined_Ts[i_camera]))
        X_triangulated = triangulate_3D_point_DLT(P, x_pairs[i_camera])
        plot_3d_points_and_cameras(X_triangulated, P)


if __name__ == "__main__":
    main()
    