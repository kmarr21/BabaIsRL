#!/bin/bash

# monitor_neuro_symbolic.sh
# Script to check the status of the neurosymbolic experiment

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

# Find the most recent log file
EXP_LOG=$(ls -t logs/neuro_symbolic_*.log 2>/dev/null | head -n 1)

# Show recent activity from log file
if [ -n "$EXP_LOG" ]; then
    echo -e "\nRecent activity from the experiment:"
    tail -n 20 "$EXP_LOG"
    
    # Count configurations completed
    echo -e "\nProgress summary:"
    total_configs=$((6 * 2 * 5))  # 6 templates * 2 models * 5 seeds
    
    completed=$(grep -c "Successfully completed:" "$EXP_LOG")
    failed=$(grep -c "All retries failed for:" "$EXP_LOG")
    
    echo "Total configurations: $total_configs"
    echo "Completed: $completed"
    echo "Failed: $failed"
    echo "Progress: $(( (completed * 100) / total_configs ))%"
fi

echo -e "\nTo see more details, use: tail -f $EXP_LOG"
