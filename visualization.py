import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from utils import *

COLORS = [
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


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Visualize 3D-Reconstruction.")
    parser.add_argument("dataset", type=str, help="Name of the dataset")
    return parser.parse_args()


def visualize_results():
    """Load and visualize the results from the Structure-from-Motion pipeline."""

    args = parse_args()

    # Load the results from the file
    with open(f"results/dataset_{args.dataset}/sfm_results.pkl", "rb") as f:
        data = pickle.load(f)

        # Extract necessary data
    K = data["K"]
    absolute_rotations = data["absolute_rotations"]
    refined_Ts = data["refined_Ts"]
    x_pairs = data["x_pairs"]
    nr_images = data["nr_images"]

    # Call the utility function to visualize the results
    visualize_sfm_results(K, absolute_rotations, refined_Ts, x_pairs, nr_images, COLORS)


if __name__ == "__main__":
    visualize_results()
