#!/bin/bash

# run Experiment 2 with 6K episodes

echo "Starting Experiment 2 (6K): KSM Comparison"
echo "====================================="

# templates to run
TEMPLATES=("basic_med" "sparse_med" "zipper_med" "bottleneck_med" "bottleneck_hard" "corridors_med")

# seeds to use
SEEDS=(42 101 202 303 404)

# directory for experiment results
EXP_DIR="experiments/experiment2_6k"

# run a single configuration with retry mechanism
run_with_retry() {
    local template=$1
    local config_name=$2
    local ksm_mode=$3
    local seed=$4
    local output_dir="${EXP_DIR}/template_${template}/${config_name}/seed${seed}"
    local max_retries=3
    local retry_count=0

    mkdir -p "$output_dir"
    
    # build command (all configs use reward shaping and state augmentation) (6k episodes)
    cmd="python train_enhanced_dqn.py --template $template --episodes 6000 --output $output_dir --seed $seed"
    
    # add KSM mode if specified
    if [ "$ksm_mode" != "off" ]; then
        cmd="${cmd} --ksm-mode ${ksm_mode}"
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
        # full enhanced DQN (reward shaping + state augmentation, no KSM)
        run_with_retry "$template" "full_enhanced_dqn" "off" "$seed"
        
        # full enhanced DQN + standard KSM
        run_with_retry "$template" "standard_ksm" "standard" "$seed"
        
        # full enhanced DQN + adaptive KSM
        run_with_retry "$template" "adaptive_ksm" "adaptive" "$seed"
    done
done

echo ""
echo "Experiment 2 (6K) completed!"
