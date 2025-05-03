#!/bin/bash

# monitor_experiments.sh
# Script to check the status of running experiments

# Check if experiment PIDs exist
if [ ! -f logs/experiment1_pid.txt ] || [ ! -f logs/experiment2_pid.txt ]; then
    echo "Experiment PID files not found. Experiments may not be running."
    exit 1
fi

EXP1_PID=$(cat logs/experiment1_pid.txt)
EXP2_PID=$(cat logs/experiment2_pid.txt)

# Function to check if a process is running
is_running() {
    if ps -p "$1" > /dev/null; then
        return 0  # Process is running
    else
        return 1  # Process is not running
    fi
}

# Check status
if is_running "$EXP1_PID"; then
    echo "Experiment 1 is running (PID: $EXP1_PID)"
else
    echo "Experiment 1 is not running (PID was: $EXP1_PID)"
fi

if is_running "$EXP2_PID"; then
    echo "Experiment 2 is running (PID: $EXP2_PID)"
else
    echo "Experiment 2 is not running (PID was: $EXP2_PID)"
fi

# Find the most recent log files
EXP1_LOG=$(ls -t logs/experiment1_*.log 2>/dev/null | head -n 1)
EXP2_LOG=$(ls -t logs/experiment2_*.log 2>/dev/null | head -n 1)

# Show recent activity from log files
if [ -n "$EXP1_LOG" ]; then
    echo -e "\nRecent activity from Experiment 1:"
    tail -n 10 "$EXP1_LOG"
fi

if [ -n "$EXP2_LOG" ]; then
    echo -e "\nRecent activity from Experiment 2:"
    tail -n 10 "$EXP2_LOG"
fi

echo -e "\nTo see more details, use: tail -f <log_file>"
