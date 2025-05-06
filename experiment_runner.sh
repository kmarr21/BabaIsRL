#!/bin/bash

# experiment_runner.sh
# MAIN script to run all experiments in the background!

# create directory structure (in case not there)
mkdir -p experiments/experiment1 experiments/experiment2 logs plots/experiment1 plots/experiment2

# record start time
START_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="logs/experiment_runner_${START_TIME}.log"

# run experiment 1 and experiment 2 in background (so can shut down SSH terminal as long as VM instance is on)
nohup bash run_experiment1.sh > logs/experiment1_${START_TIME}.log 2>&1 &
EXP1_PID=$!
echo "Started Experiment 1 with PID: $EXP1_PID"
echo "Experiment 1 log file: logs/experiment1_${START_TIME}.log"

# wait a bit before starting Exp2 to avoid initial resource contention
sleep 10

nohup bash run_experiment2.sh > logs/experiment2_${START_TIME}.log 2>&1 &
EXP2_PID=$!
echo "Started Experiment 2 with PID: $EXP2_PID"
echo "Experiment 2 log file: logs/experiment2_${START_TIME}.log"

# save PIDs to files for reference
echo $EXP1_PID > logs/experiment1_pid.txt
echo $EXP2_PID > logs/experiment2_pid.txt

echo "All experiments started. You can safely close the SSH session."
echo "To check experiment status: tail -f logs/experiment1_${START_TIME}.log logs/experiment2_${START_TIME}.log"
echo "To generate plots after completion: bash generate_plots.sh"
