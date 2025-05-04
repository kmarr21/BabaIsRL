#!/bin/bash

# monitor_experiment.sh
# Script to monitor the progress of the neurosymbolic experiment

# Base directories
BASE_DIR="experiments/neuro_symbolic_comparison"
LOG_DIR="logs"

# Templates
TEMPLATES=("basic_med" "sparse_med" "zipper_med" "bottleneck_med" "bottleneck_hard" "corridors_med")

# Models
MODELS=("enhanced_dqn" "neurosymbolic_dqn")

# Seeds
SEEDS=(42 101 202 303 404)

echo "===========================================" 
echo "Neurosymbolic Experiment Progress Monitor"
echo "===========================================" 
echo "Current time: $(date)"
echo

# Check CPU and Memory usage
echo "System Resources:"
echo "-----------------"
echo "CPU Usage:"
top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}'
echo "Memory Usage:"
free -h | grep "Mem:" | awk '{print "Total: " $2 "  Used: " $3 "  Free: " $4}'
echo

# Check running processes
echo "Running Experiment Processes:"
echo "----------------------------"
ps aux | grep -E "train_enhanced_dqn.py|train_neuro_symbolic.py" | grep -v grep
echo

# Check experiment progress
echo "Experiment Progress Summary:"
echo "---------------------------"
total_runs=$((${#TEMPLATES[@]} * ${#MODELS[@]} * ${#SEEDS[@]}))
completed_runs=0
running_runs=0

for template in "${TEMPLATES[@]}"; do
    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            output_dir="$BASE_DIR/$template/$model/seed$seed"
            
            # Check if directory exists
            if [ -d "$output_dir" ]; then
                # Check if there's a PID file
                if [ -f "$output_dir/pid.txt" ]; then
                    pid=$(cat "$output_dir/pid.txt")
                    
                    # Check if process is running
                    if ps -p "$pid" > /dev/null; then
                        # Process is running
                        running_runs=$((running_runs + 1))
                        
                        # Get progress from log
                        log_file=$(ls -t "$LOG_DIR/${template}_${model}_seed${seed}_"*.log 2>/dev/null | head -n 1)
                        if [ -n "$log_file" ]; then
                            episode=$(grep -o -E "Episode [0-9]+/[0-9]+" "$log_file" | tail -n 1)
                            status="RUNNING - $episode"
                        else
                            status="RUNNING - No log found"
                        fi
                    else
                        # Process is not running, check if completed
                        if [ -f "$output_dir/dqn_final.pth" ]; then
                            status="COMPLETED"
                            completed_runs=$((completed_runs + 1))
                        else
                            # Check if there's a checkpoint file
                            checkpoint_files=$(ls -t "$output_dir/dqn_checkpoint_"*.pth 2>/dev/null | head -n 1)
                            if [ -n "$checkpoint_files" ]; then
                                checkpoint=$(basename "$checkpoint_files" | grep -o -E '[0-9]+' | head -n 1)
                                status="STOPPED at episode $checkpoint"
                            else
                                status="FAILED - No checkpoint found"
                            fi
                        fi
                    fi
                else
                    # No PID file, check if completed
                    if [ -f "$output_dir/dqn_final.pth" ]; then
                        status="COMPLETED"
                        completed_runs=$((completed_runs + 1))
                    else
                        # Check if there's a checkpoint file
                        checkpoint_files=$(ls -t "$output_dir/dqn_checkpoint_"*.pth 2>/dev/null | head -n 1)
                        if [ -n "$checkpoint_files" ]; then
                            checkpoint=$(basename "$checkpoint_files" | grep -o -E '[0-9]+' | head -n 1)
                            status="UNKNOWN at episode $checkpoint"
                        else
                            status="NOT STARTED"
                        fi
                    fi
                fi
                
                printf "%-12s %-18s Seed: %-5s Status: %s\n" "$template" "$model" "$seed" "$status"
            else
                printf "%-12s %-18s Seed: %-5s Status: NOT STARTED\n" "$template" "$model" "$seed"
            fi
        done
    done
done

# Print summary
echo
echo "Summary:"
echo "--------"
echo "Total runs: $total_runs"
echo "Completed: $completed_runs"
echo "Running: $running_runs"
echo "Remaining: $((total_runs - completed_runs - running_runs))"
echo "Progress: $(( 100 * completed_runs / total_runs ))%"

# Check disk space
echo
echo "Disk Space:"
echo "-----------"
df -h | grep -E "^Filesystem|/$"

echo
echo "To see detailed logs: tail -f logs/*.log"
echo "To generate plots after completion: bash generate_neuro_symbolic_plots.sh"
echo
echo "Last updated: $(date)"
