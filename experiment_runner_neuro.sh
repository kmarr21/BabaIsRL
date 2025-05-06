#!/bin/bash

# experiment_runner_neuro.sh
# run neurosymbolic experiment in the background

# create directory structure
mkdir -p experiments/neuro_symbolic experiments/neuro_symbolic_comparison logs plots/neuro_symbolic

# record start time
START_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="logs/experiment_runner_${START_TIME}.log"

# run NS experiment in background
nohup bash run_neuro_symbolic.sh > logs/neuro_symbolic_${START_TIME}.log 2>&1 &
EXP_PID=$!
echo "Started Neurosymbolic Experiment with PID: $EXP_PID"
echo "Experiment log file: logs/neuro_symbolic_${START_TIME}.log"

# save PID to file for reference
echo $EXP_PID > logs/neuro_symbolic_pid.txt

echo "All experiments started. You can safely close the SSH session."
echo "To check experiment status: tail -f logs/neuro_symbolic_${START_TIME}.log"
echo "To generate plots after completion: bash generate_neuro_symbolic_plots.sh"
