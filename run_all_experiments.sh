#!/bin/bash

# Set up experiments to run
TEMPLATES=("sparse_med" "zipper_med" "bottleneck_hard")
EPISODES=2000
SEEDS=5

# Create log directory
mkdir -p experiment_logs

# Function to run an experiment with caffeinate
run_experiment() {
    EXPERIMENT=$1
    TEMPLATE=$2
    
    echo "Starting $EXPERIMENT on $TEMPLATE..."
    
    # Run command with caffeinate and log output
    caffeinate -i python run_experiment.py \
        --experiment $EXPERIMENT \
        --template $TEMPLATE \
        --episodes $EPISODES \
        --seeds $SEEDS \
        | tee experiment_logs/${EXPERIMENT}_${TEMPLATE}.log
        
    echo "Completed $EXPERIMENT on $TEMPLATE"
}

# Run all experiment combinations
for TEMPLATE in "${TEMPLATES[@]}"; do
    run_experiment "dqn_variants" $TEMPLATE
    run_experiment "ksm_variants" $TEMPLATE
    run_experiment "ksm_with_baselines" $TEMPLATE
done

echo "All experiments completed!"
