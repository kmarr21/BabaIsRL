#!/bin/bash

# setup_experiment.sh
# Script to install dependencies and set up the experiment environment

echo "Setting up neurosymbolic experiment environment"
echo "==============================================="

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install it before continuing."
    exit 1
fi

# Check for pip
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed. Please install it before continuing."
    exit 1
fi

# Install required packages
echo "Installing required Python packages..."
pip3 install numpy torch matplotlib pandas scipy tqdm gymnasium pygame

# Create directory structure
echo "Creating directory structure..."
mkdir -p experiments/neuro_symbolic_comparison
mkdir -p logs
mkdir -p ~/neuro_symbolic_plots

# Make scripts executable
echo "Making scripts executable..."
chmod +x experiment_neuro_symbolic.sh
chmod +x generate_neuro_symbolic_plots.sh
chmod +x monitor_experiment.sh

echo "Setup complete!"
echo
echo "To start the experiment: ./experiment_neuro_symbolic.sh"
echo "To monitor progress: ./monitor_experiment.sh"
echo "To generate plots after completion: ./generate_neuro_symbolic_plots.sh"
