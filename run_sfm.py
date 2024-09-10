import os
import argparse
import logging
import pickle
import numpy as np
from utils import *
from src import (
    process_sift_for_image_pairs,
    estimate_R,
    run_reconstruction,
    estimate_translation,
    refine_translation,
)


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run Structure-from-Motion pipeline.")
    parser.add_argument("data_path", type=str, help="Base path to the dataset")
    parser.add_argument("dataset", type=str, help="Name of the dataset")
    parser.add_argument(
        "threshold", type=float, default=1.0, help="Pixel threshold for matching"
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: CRITICAL)",
    )
    return parser.parse_args()


def run_sfm():
    """Main function to run the structure from motion pipeline."""
    args = parse_args()
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

    # Refine each T using Levenberg-Marquardt algorithm
    refined_Ts = refine_translation(
        K, img_paths, desc_X_inliers, X0, absolute_rotations, estimated_Ts
    )

    # Save results
    output_dir = f"results/dataset_{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    with open(f"{output_dir}/sfm_results.pkl", "wb") as f:
        pickle.dump(
            {
                "K": K,
                "absolute_rotations": absolute_rotations,
                "refined_Ts": refined_Ts,
                "x_pairs": x_pairs,
                "nr_images": nr_images,
            },
            f,
        )

    logging.info("Structure-from-Motion pipeline completed and results saved.")


if __name__ == "__main__":
    run_sfm()
