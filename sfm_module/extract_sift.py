import cv2
import numpy as np
import logging
from tqdm import tqdm
import time
from auxiliary import *


def process_sift_for_image_pairs(img_paths, init_pair, dataset):
    """
    Extracts SIFT data for image pairs and an initial pair.
    - First extracts SIFT pairs for consecutive images.
    - Then extracts SIFT features for an initial pair of images.
    If the data already exists, it loads from the saved files.

    Parameters:
    - img_names: List of image file names
    - init_pair: Initial pair of indices for image matching
    - dataset: Name of the dataset

    Returns:
    - x_pairs: List of extracted SIFT pairs
    - initial_pair: Extracted SIFT data for the initial pair
    """

    # Define filenames for saving/loading SIFT data
    x_pairs_filename = f"x_pairs_dataset_{dataset}.pkl"
    init_pair_filename = f"init_pair_dataset_{dataset}.pkl"

    # Attempt to load x_pairs data
    x_pairs = load_x_pairs(x_pairs_filename)
    if x_pairs is None:
        x_pairs = []
        nr_images = len(img_paths)
        # Process consecutive image pairs
        for i in tqdm(range(nr_images - 1), desc="Extracting SIFT pairs"):
            start_time = time.time()
            x1, x2, _ = compute_sift_keypoints(img_paths[i], img_paths[i + 1])
            elapsed_time = time.time() - start_time
            logging.info(
                f"SIFT extraction completed for pair {i}-{i+1} in {elapsed_time:.2f} seconds"
            )
            x_pairs.extend([x1, x2])

        # Save the extracted SIFT pairs
        save_x_pairs(x_pairs, x_pairs_filename)
        logging.info(f"x_pairs saved to {x_pairs_filename}")
    else:
        logging.info(f"x_pairs loaded from {x_pairs_filename}")

    # Attempt to load the initial pair data
    initial_pair = load_x_pairs(init_pair_filename)
    if initial_pair is None:
        start_time = time.time()
        init_imgs = [img_paths[i] for i in init_pair]
        init_x1, init_x2, desc_X = compute_sift_keypoints(init_imgs[0], init_imgs[1])
        elapsed_time = time.time() - start_time

        logging.info(f"Initial SIFT extraction completed in {elapsed_time:.2f} seconds")
        initial_pair = [init_x1, init_x2, desc_X]

        # Save the extracted initial pair
        save_x_pairs(initial_pair, init_pair_filename)
        logging.info(f"Initial pair saved to {init_pair_filename}")
    else:
        logging.info(f"Initial pair loaded from {init_pair_filename}")

    return x_pairs, initial_pair


def compute_sift_keypoints(image_path1: str, image_path2: str):
    """
    Extracts and matches SIFT keypoints between two images.

    Args:
        image_path1 (str): Path to the first image.
        image_path2 (str): Path to the second image.

    Returns:
        tuple:
            - x1 (np.ndarray): Homogeneous coordinates of matched keypoints from the first image (3xN).
            - x2 (np.ndarray): Homogeneous coordinates of matched keypoints from the second image (3xN).
            - desc_X1 (np.ndarray): Descriptors of the matched keypoints from the first image (128xN).

    """
    # Load images
    im1 = cv2.imread(image_path1)
    im2 = cv2.imread(image_path2)
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect SIFT features and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Match descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good_matches.append(m)

    # Extract coordinates of matched keypoints
    x1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).T
    x2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).T

    x1 = cartesian_to_homogeneous(x1)
    x2 = cartesian_to_homogeneous(x2)

    # Save descriptors
    desc_X1 = np.array([descriptors1[m.queryIdx] for m in good_matches]).T

    return x1, x2, desc_X1
