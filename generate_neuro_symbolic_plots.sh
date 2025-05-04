#!/bin/bash

# generate_neuro_symbolic_plots.sh
# Script to generate aggregated plots from neurosymbolic experiment results

echo "Generating plots for neurosymbolic experiment..."

# Run the Python script to generate plots
python plot_neuro_symbolic_results.py

echo "Plots generated in plots/neuro_symbolic directory"
