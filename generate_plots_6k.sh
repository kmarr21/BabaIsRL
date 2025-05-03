#!/bin/bash

# generate_plots_6k.sh
# Script to generate aggregated plots from 6K experiment results

# Default to generating plots for both experiments if none specified
EXPERIMENTS=("experiment1_6k" "experiment2_6k")
if [ $# -ge 1 ]; then
    EXPERIMENTS=("$@")
fi

echo "Generating plots for experiments: ${EXPERIMENTS[*]}"

# Run the Python script to generate plots
python new_plot_results.py "${EXPERIMENTS[@]}"

echo "Plots generated in plots/ directory"

# Create a directory to store the plots for downloading
mkdir -p ~/plot_chunks_6k

# Zip up the plot directories
for experiment in "${EXPERIMENTS[@]}"; do
    for template in basic_med sparse_med zipper_med bottleneck_med bottleneck_hard corridors_med; do
        zip -r ~/plot_chunks_6k/${experiment}_${template}.zip plots/${experiment}/${template}/
    done
done

echo "Plot archives created in ~/plot_chunks_6k/"
echo "You can now download these files individually"
