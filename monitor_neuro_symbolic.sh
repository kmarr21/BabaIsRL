#!/bin/bash

# monitor_neuro_symbolic.sh
# Script to check the status of running neurosymbolic experiment

# Check if experiment PID exists
if [ ! -f logs/neuro_symbolic_pid.txt ]; then
    echo "Experiment PID file not found. Experiment may not be running."
    exit 1
fi

EXP_PID=$(cat logs/neuro_symbolic_pid.txt)

# Function to check if a process is running
is_running() {
    if ps -p "$1" > /dev/null; then
        return 0  # Process is running
    else
        return 1  # Process is not running
    fi
}

# Check status
if is_running "$EXP_PID"; then
    echo "Neurosymbolic experiment is running (PID: $EXP_PID)"
else
    echo "Neurosymbolic experiment is not running (PID was: $EXP_PID)"
fi

# Find the most recent log files
EXP_LOG=$(ls -t logs/neuro_symbolic_*.log 2>/dev/null | head -n 1)

# Show recent activity from log files
if [ -n "$EXP_LOG" ]; then
    echo -e "\nRecent activity from Neurosymbolic Experiment:"
    tail -n 20 "$EXP_LOG"
fi

echo -e "\nTo see more details, use: tail -f <log_file>"
