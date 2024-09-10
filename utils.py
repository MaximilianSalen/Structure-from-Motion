import os
import pickle
import time
import logging
import yaml


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
