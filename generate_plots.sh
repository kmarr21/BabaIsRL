#!/bin/bash

# generate_plots.sh
# generates aggregated plots from experiment results

# default to generating plots for both experiments if none specified
EXPERIMENTS=("experiment1" "experiment2")
if [ $# -ge 1 ]; then
    EXPERIMENTS=("$@")
fi

echo "Generating plots for experiments: ${EXPERIMENTS[*]}"

# run the Python script to generate plots
python new_plot_results.py "${EXPERIMENTS[@]}"

echo "Plots generated in plots/ directory"
