import argparse
import pickle
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
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    return parser.parse_args()


def visualize_results():
    """Load and visualize the results from the Structure-from-Motion pipeline."""

    args = parse_args()

    # Load the results from the file
    with open(f"results/dataset_{args.dataset}/sfm_results.pkl", "rb") as f:
        sfm_data_dict = pickle.load(f)

    # Visualize and animate the SfM results
    visualize_sfm_results_with_rotation(
        sfm_data_dict, dataset_name=args.dataset, colors=COLORS
    )


if __name__ == "__main__":
    visualize_results()
