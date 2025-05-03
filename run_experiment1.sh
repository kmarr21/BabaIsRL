#!/bin/bash

# run_experiment1.sh
# Script to run Experiment 1: Ablation Study

echo "Starting Experiment 1: Ablation Study"
echo "======================================"

# Templates to run
TEMPLATES=("basic_med" "sparse_med" "zipper_med" "bottleneck_med" "bottleneck_hard" "corridors_med")

# Seeds to use
SEEDS=(42 101 202 303 404)

# Directory for experiment results
EXP_DIR="experiments/experiment1"

# Function to run a single configuration with retry mechanism
run_with_retry() {
    local template=$1
    local config_name=$2
    local reward_shaping=$3
    local state_aug=$4
    local ksm_mode=$5
    local seed=$6
    local output_dir="${EXP_DIR}/template_${template}/${config_name}/seed${seed}"
    local max_retries=3
    local retry_count=0

    mkdir -p "$output_dir"
    
    # Build command
    cmd="python train_enhanced_dqn.py --template $template --episodes 4000 --output $output_dir --seed $seed"
    
    # Add configuration options
    if [ "$reward_shaping" = false ]; then
        cmd="${cmd} --no-reward-shaping"
    fi
    
    if [ "$state_aug" = false ]; then
        cmd="${cmd} --basic-state"
    fi
    
    if [ "$ksm_mode" != "off" ]; then
        cmd="${cmd} --ksm-mode ${ksm_mode}"
    fi
    
    echo "Running: $cmd"
    
    # Retry loop
    while [ $retry_count -lt $max_retries ]; do
        # Run the command
        $cmd > "${output_dir}/training.log" 2>&1
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo "✓ Successfully completed: ${template}/${config_name}/seed${seed}"
            return 0
        else
            retry_count=$((retry_count + 1))
            echo "✗ Failed: ${template}/${config_name}/seed${seed} (Attempt ${retry_count}/${max_retries})"
            sleep 10  # Wait before retrying
        fi
    done
    
    echo "✗✗✗ All retries failed for: ${template}/${config_name}/seed${seed}"
    return 1
}

# Run experiments for each template
for template in "${TEMPLATES[@]}"; do
    echo ""
    echo "Starting template: $template"
    echo "------------------------"
    
    # Create template directory
    mkdir -p "${EXP_DIR}/template_${template}"
    
    # Run each configuration with all seeds
    for seed in "${SEEDS[@]}"; do
        # Baseline DQN (no reward shaping, no state aug)
        run_with_retry "$template" "baseline_dqn" false false "off" "$seed"
        
        # DQN with only reward shaping
        run_with_retry "$template" "reward_shaping_only" true false "off" "$seed"
        
        # DQN with only state augmentation
        run_with_retry "$template" "state_aug_only" false true "off" "$seed"
        
        # Full enhanced DQN (reward shaping + state augmentation)
        run_with_retry "$template" "full_enhanced_dqn" true true "off" "$seed"
    done
done

echo ""
echo "Experiment 1 completed!"
echo "To generate plots, run: bash generate_plots.sh experiment1"
