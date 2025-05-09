#!/bin/bash

# generates aggregated plots from neurosymbolic experiment results

echo "Generating plots for neurosymbolic experiment..."

# run python script to generate plots
python plot_neuro_symbolic_results.py

echo "Plots generated in plots/neuro_symbolic directory"
