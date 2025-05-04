#!/bin/bash

# experiment_neuro_symbolic.sh
# Script to run the neurosymbolic DQN experiment in the background
# Resistant to SSH disconnects and other interruptions

# Create directory structure for results
BASE_DIR="experiments/neuro_symbolic_comparison"
PLOTS_DIR="~/neuro_symbolic_plots"  # Directory outside the repo
LOG_DIR="logs"

# Create directories
mkdir -p $BASE_DIR $LOG_DIR
mkdir -p "$PLOTS_DIR"

# Templates to run
TEMPLATES=("basic_med" "sparse_med" "zipper_med" "bottleneck_med" "bottleneck_hard" "corridors_med")

# Seeds to use
SEEDS=(42 101 202 303 404)

# Episode count
EPISODES=5000

# Record start time
START_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
MAIN_LOG="$LOG_DIR/experiment_main_${START_TIME}.log"

echo "Starting Neurosymbolic DQN Comparison Experiment at $START_TIME" | tee -a $MAIN_LOG
echo "Running with templates: ${TEMPLATES[*]}" | tee -a $MAIN_LOG
echo "Seeds: ${SEEDS[*]}" | tee -a $MAIN_LOG
echo "Episodes per run: $EPISODES" | tee -a $MAIN_LOG
echo "=========================" | tee -a $MAIN_LOG

# Function to run a single configuration with error handling and logging
run_config() {
    local template=$1
    local seed=$2
    local model_type=$3
    local output_dir="$BASE_DIR/$template/$model_type/seed$seed"
    local log_file="$LOG_DIR/${template}_${model_type}_seed${seed}_${START_TIME}.log"
    
    # Create the output directory
    mkdir -p $output_dir
    
    echo "Starting run: Template=$template, Seed=$seed, Model=$model_type" | tee -a $MAIN_LOG
    echo "Log file: $log_file" | tee -a $MAIN_LOG
    
    # Build command based on model type
    if [ "$model_type" = "enhanced_dqn" ]; then
        # Enhanced DQN baseline - using train_enhanced_dqn.py
        cmd="python train_enhanced_dqn.py --template $template --seed $seed --episodes $EPISODES --output $output_dir"
    else
        # Neurosymbolic DQN with gradual guidance decrease
        cmd="python train_neuro_symbolic.py --template $template --seed $seed --episodes $EPISODES --gradual-guidance --output $output_dir"
    fi
    
    echo "Running command: $cmd" | tee -a $MAIN_LOG
    
    # Run the command in background with nohup and redirect output to log file
    nohup $cmd > $log_file 2>&1 &
    local pid=$!
    echo "Process started with PID: $pid" | tee -a $MAIN_LOG
    
    # Save PID to a file for monitoring
    echo $pid > "$output_dir/pid.txt"
    
    # Wait for process to complete
    wait $pid
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ Successfully completed: Template=$template, Seed=$seed, Model=$model_type" | tee -a $MAIN_LOG
    else
        echo "✗ Failed: Template=$template, Seed=$seed, Model=$model_type (Exit code: $exit_code)" | tee -a $MAIN_LOG
    fi
    
    return $exit_code
}

# Run all configurations in sequence
for template in "${TEMPLATES[@]}"; do
    echo "Starting template: $template" | tee -a $MAIN_LOG
    echo "------------------------" | tee -a $MAIN_LOG
    
    for seed in "${SEEDS[@]}"; do
        # Run Enhanced DQN
        run_config "$template" "$seed" "enhanced_dqn"
        
        # Run Neurosymbolic DQN
        run_config "$template" "$seed" "neurosymbolic_dqn"
    done
    
    echo "Completed template: $template" | tee -a $MAIN_LOG
    echo "" | tee -a $MAIN_LOG
done

echo "All experiment runs have been scheduled!" | tee -a $MAIN_LOG
echo "To monitor progress, check: $MAIN_LOG" | tee -a $MAIN_LOG
echo "To generate plots after completion, run: bash generate_neuro_symbolic_plots.sh" | tee -a $MAIN_LOG
