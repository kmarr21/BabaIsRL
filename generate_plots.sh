#!/bin/bash

# generate_plots.sh
# Script to generate aggregated plots from experiment results

# Default to generating plots for both experiments if none specified
EXPERIMENTS=("experiment1" "experiment2")
if [ $# -ge 1 ]; then
    EXPERIMENTS=("$@")
fi

echo "Generating plots for experiments: ${EXPERIMENTS[*]}"

# Run the Python script to generate plots
python plot_results.py "${EXPERIMENTS[@]}"

echo "Plots generated in plots/ directory"
