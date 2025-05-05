#!/bin/bash

# run_neurosymbolic_plots.sh
# Script to generate plots for neurosymbolic experiments

# Create a fresh directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FRESH_DIR=~/neurosymbolic_plots_${TIMESTAMP}

echo "Creating plots in: ${FRESH_DIR}"
mkdir -p ${FRESH_DIR}

# Run the Python script
python ~/plot_neurosymbolic_experiments.py --output-dir ${FRESH_DIR} --data-dir ~/BabaIsRL/experiments

echo "Plots generated in: ${FRESH_DIR}"
echo "Zip archives available in the same directory"
