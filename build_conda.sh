#!/bin/bash
NAME="sfm_venv"

# Locate path of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Create the conda environment if it doesn't exist
if ! conda info --envs | grep -q "$NAME"; then
  conda create -n $NAME python=3.12 -y
fi

# Use conda run to install the required packages in the environment
conda run -n $NAME pip install -r "$DIR/requirements.txt"

echo "Conda environment $NAME is ready and the required packages are installed."
