#!/bin/bash
NAME="sfm_venv"

# locate path of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

conda create -n $NAME python=3.12 -y
conda activate $NAME

# Use conda run to install packages inside the environment
conda run -n $NAME pip -r requirements.txt