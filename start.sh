#!/usr/bin/env bash

# Default values for address and port
SERVER_ADDRESS="127.0.0.1"
SERVER_PORT="8501"

# Parse command-line arguments
while getopts "a:p:" opt; do
  case $opt in
    a) SERVER_ADDRESS="$OPTARG"
    ;;
    p) SERVER_PORT="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1
    ;;
  esac
done

# Set up environment variables
CONDA_ROOT_PREFIX="$(pwd)/installer_files/conda"
INSTALL_ENV_DIR="$(pwd)/installer_files/env"

# Activate installer environment
source "$CONDA_ROOT_PREFIX/etc/profile.d/conda.sh" # Initialize conda for use in the script
conda activate "$INSTALL_ENV_DIR"

# Run Streamlit app with the specified address and port
streamlit run app.py --server.address "$SERVER_ADDRESS" --server.port "$SERVER_PORT"
# streamlit hello