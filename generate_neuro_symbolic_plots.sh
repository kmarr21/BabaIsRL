#!/bin/bash

# generate_neuro_symbolic_plots.sh
# Script to generate comparison plots for the neurosymbolic DQN experiment

# Base directories
BASE_DIR="experiments/neuro_symbolic_comparison"
PLOT_DIR="~/neuro_symbolic_plots"
LOG_DIR="logs"

# Templates
TEMPLATES=("basic_med" "sparse_med" "zipper_med" "bottleneck_med" "bottleneck_hard" "corridors_med")

# Create the plots directory if it doesn't exist
mkdir -p "$PLOT_DIR"

# Record start time
START_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$LOG_DIR/plot_generation_${START_TIME}.log"

echo "Starting Neurosymbolic DQN Plot Generation at $START_TIME" | tee -a $LOG_FILE
echo "Saving plots to: $PLOT_DIR" | tee -a $LOG_FILE
echo "=========================" | tee -a $LOG_FILE

# Generate plots using the plot script
echo "Running plot generation script..." | tee -a $LOG_FILE
python plot_neuro_symbolic_comparison.py --experiment-dir $BASE_DIR --output-dir $PLOT_DIR | tee -a $LOG_FILE

# Check if plot generation was successful
if [ $? -eq 0 ]; then
    echo "Plot generation completed successfully!" | tee -a $LOG_FILE
    
    # List all generated plots
    echo "Generated plots:" | tee -a $LOG_FILE
    ls -la $PLOT_DIR | grep -E '\.png$' | tee -a $LOG_FILE
    
    # Create a tarball of all plots for easy download
    tar_file="$PLOT_DIR/neuro_symbolic_plots_${START_TIME}.tar.gz"
    tar -czvf $tar_file -C $PLOT_DIR ./*.png
    
    echo "Created archive of all plots: $tar_file" | tee -a $LOG_FILE
    echo "You can download this file using SCP or SFTP" | tee -a $LOG_FILE
    echo "Example: scp username@your-vm-ip:$tar_file /local/path/" | tee -a $LOG_FILE
else
    echo "Plot generation encountered errors. Check the log for details." | tee -a $LOG_FILE
fi

echo "Plot generation process completed at $(date)" | tee -a $LOG_FILE
