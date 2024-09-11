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
    parser.add_argument("dataset", type=str, help="Name of the dataset")
    parser.add_argument("--save_path", type=str, help="Path to save the animated GIF")
    parser.add_argument(
        "--tag",
        type=str,
        default="sfm_rotation",
        help="Tag for the saved file name (default: 'sfm_rotation')",
    )
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

    # Visualize and animate the SfM results
    visualize_sfm_results_with_rotation(
        K=K,
        absolute_rotations=absolute_rotations,
        refined_Ts=refined_Ts,
        x_pairs=x_pairs,
        nr_images=nr_images,
        colors=COLORS,
        save_path=args.save_path,
        tag=args.tag,
    )


if __name__ == "__main__":
    visualize_results()
