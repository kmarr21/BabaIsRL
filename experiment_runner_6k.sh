#!/bin/bash

# experiment_runner_6k.sh
# run all 6K episodes experiments in the background!

# create directory structure (in case not there)
mkdir -p experiments/experiment1_6k experiments/experiment2_6k logs plots/experiment1_6k plots/experiment2_6k

# record start time
START_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="logs/experiment_runner_6k_${START_TIME}.log"

# run Experiment 1 and Experiment 2 in background
nohup bash run_experiment1_6k.sh > logs/experiment1_6k_${START_TIME}.log 2>&1 &
EXP1_PID=$!
echo "Started Experiment 1 (6K) with PID: $EXP1_PID"
echo "Experiment 1 (6K) log file: logs/experiment1_6k_${START_TIME}.log"

# wait a bit before starting Exp2 to avoid initial resource contention
sleep 10

nohup bash run_experiment2_6k.sh > logs/experiment2_6k_${START_TIME}.log 2>&1 &
EXP2_PID=$!
echo "Started Experiment 2 (6K) with PID: $EXP2_PID"
echo "Experiment 2 (6K) log file: logs/experiment2_6k_${START_TIME}.log"

# save PIDs to files for reference
echo $EXP1_PID > logs/experiment1_6k_pid.txt
echo $EXP2_PID > logs/experiment2_6k_pid.txt

echo "All 6K experiments started. You can safely close the SSH session."
echo "To check experiment status: tail -f logs/experiment1_6k_${START_TIME}.log logs/experiment2_6k_${START_TIME}.log"
