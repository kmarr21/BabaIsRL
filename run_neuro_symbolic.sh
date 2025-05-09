#!/bin/bash

# run neurosymbolic experiment comparing enhanced DQN vs NDT-DQN

echo "Starting Neurosymbolic DQN Experiment"
echo "====================================="

# templates to run
TEMPLATES=("basic_med" "sparse_med" "zipper_med" "bottleneck_med" "bottleneck_hard" "corridors_med")

# seeds to use
SEEDS=(42 101 202 303 404)

# directory for experiment results
EXP_DIR="experiments/neuro_symbolic"

# function to run a single configuration w/ retry mechanism
run_with_retry() {
    local template=$1
    local config_name=$2
    local use_neuro_symbolic=$3
    local seed=$4
    local output_dir="${EXP_DIR}/template_${template}/${config_name}/seed${seed}"
    local max_retries=3
    local retry_count=0

    mkdir -p "$output_dir"
    
    # build command
    if [ "$use_neuro_symbolic" = true ]; then
        # Neurosymbolic DQN with gradual guidance decrease
        cmd="python train_neuro_symbolic.py --template $template --episodes 5000 --output $output_dir --seed $seed --gradual-guidance"
    else
        # enhanced DQN baseline 
        cmd="python train_enhanced_dqn.py --template $template --episodes 5000 --output $output_dir --seed $seed"
    fi
    
    echo "Running: $cmd"
    
    # retry loop
    while [ $retry_count -lt $max_retries ]; do
        # run command
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

# run experiments for each template
for template in "${TEMPLATES[@]}"; do
    echo ""
    echo "Starting template: $template"
    echo "------------------------"
    
    # create template directory
    mkdir -p "${EXP_DIR}/template_${template}"
    
    # run each configuration with all seeds
    for seed in "${SEEDS[@]}"; do
        # enhanced DQN (baseline)
        run_with_retry "$template" "enhanced_dqn" false "$seed"
        
        # NDT-DQN with gradual guidance
        run_with_retry "$template" "neurosymbolic_dqn" true "$seed"
    done
done

echo ""
echo "Neurosymbolic experiment completed!"
echo "To generate plots, run: bash generate_neuro_symbolic_plots.sh"
