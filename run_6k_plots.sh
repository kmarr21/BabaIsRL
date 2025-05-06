#!/bin/bash

# Create a fresh directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FRESH_DIR=~/fresh_6k_plots_${TIMESTAMP}

echo "Creating plots in: ${FRESH_DIR}"
mkdir -p ${FRESH_DIR}

# Run the Python script
python ~/plot_6k_experiments.py --output-dir ${FRESH_DIR} --data-dir ~/BabaIsRL/experiments

echo "Plots generated in: ${FRESH_DIR}"
