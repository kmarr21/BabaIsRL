#!/bin/bash

# monitor_neuro_symbolic.sh
# Simple script to monitor progress of the experiment

# Base directory
BASE_DIR="experiments/neuro_symbolic_comparison"

# Templates
TEMPLATES=("basic_med" "sparse_med" "zipper_med" "bottleneck_med" "bottleneck_hard" "corridors_med")

# Models
MODELS=("enhanced_dqn" "neurosymbolic_dqn")

# Seeds
SEEDS=(42 101 202 303 404)

echo "=========================="
echo "Neurosymbolic Experiment Status"
echo "=========================="
echo "Current time: $(date)"
echo

# Count running and completed runs
total_runs=$((${#TEMPLATES[@]} * ${#MODELS[@]} * ${#SEEDS[@]}))
running=0
completed=0
waiting=0

# Check each combination
for template in "${TEMPLATES[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "Template: $template, Model: $model"
        echo "------------------------------------"
        
        for seed in "${SEEDS[@]}"; do
            dir="$BASE_DIR/$template/$model/seed$seed"
            pid_file="$dir/pid.txt"
            
            # Check status
            if [ -f "$pid_file" ]; then
                pid=$(cat "$pid_file")
                
                if ps -p $pid > /dev/null; then
                    # Process running - check progress
                    log_files=$(ls -t logs/${template}_${model}_seed${seed}_*.log 2>/dev/null)
                    
                    if [ -n "$log_files" ]; then
                        # Get latest log file
                        log_file=$(echo "$log_files" | head -n1)
                        
                        # Extract progress from log
                        progress=$(grep -o "Episode [0-9]*/[0-9]*" "$log_file" | tail -n1)
                        
                        if [ -n "$progress" ]; then
                            status="RUNNING - $progress"
                        else
                            status="RUNNING - Starting up"
                        fi
                    else
                        status="RUNNING - No logs found"
                    fi
                    
                    running=$((running + 1))
                else
                    # Process not running, check if completed
                    if [ -f "$dir/dqn_final.pth" ]; then
                        status="COMPLETED"
                        completed=$((completed + 1))
                    else
                        status="STOPPED - Process died"
                    fi
                fi
            else
                # No PID file
                if [ -d "$dir" ]; then
                    if [ -f "$dir/dqn_final.pth" ]; then
                        status="COMPLETED"
                        completed=$((completed + 1))
                    else
                        status="WAITING"
                        waiting=$((waiting + 1))
                    fi
                else
                    status="NOT STARTED"
                    waiting=$((waiting + 1))
                fi
            fi
            
            echo "  Seed $seed: $status"
        done
        
        echo ""
    done
done

echo "Summary:"
echo "--------"
echo "Total runs: $total_runs"
echo "Running: $running"
echo "Completed: $completed"
echo "Waiting/Not Started: $waiting"
echo "Progress: $((100 * completed / total_runs))%"
echo ""
echo "To check logs: tail -f logs/*.log"
echo "To generate plots: python plot_neuro_symbolic_comparison.py"
