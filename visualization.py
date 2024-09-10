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
    with open(f"results/{args.dataset}/sfm_results.pkl", "rb") as f:
        data = pickle.load(f)
