#!/bin/bash

# experiment_neuro_symbolic.sh
# Script to run the neurosymbolic DQN experiment

# Create directory structure
mkdir -p experiments/neuro_symbolic_comparison logs plots/neuro_symbolic_comparison

# Record start time
START_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
MAIN_LOG="logs/experiment_main_${START_TIME}.log"

# Templates to run
TEMPLATES=("basic_med" "sparse_med" "zipper_med" "bottleneck_med" "bottleneck_hard" "corridors_med")

# Seeds to use
SEEDS=(42 101 202 303 404)

# Episode count
EPISODES=5000

echo "Starting Neurosymbolic DQN Comparison Experiment at $START_TIME" > $MAIN_LOG
echo "Running with templates: ${TEMPLATES[*]}" >> $MAIN_LOG
echo "Seeds: ${SEEDS[*]}" >> $MAIN_LOG
echo "Episodes per run: $EPISODES" >> $MAIN_LOG
echo "=========================" >> $MAIN_LOG

# Run Enhanced DQN and Neurosymbolic DQN in background
for template in "${TEMPLATES[@]}"; do
    echo "Starting template: $template" >> $MAIN_LOG
    echo "------------------------" >> $MAIN_LOG
    
    for seed in "${SEEDS[@]}"; do
        # Enhanced DQN
        enhanced_dir="experiments/neuro_symbolic_comparison/$template/enhanced_dqn/seed$seed"
        enhanced_log="logs/${template}_enhanced_dqn_seed${seed}_${START_TIME}.log"
        mkdir -p $enhanced_dir
        
        echo "Starting Enhanced DQN: Template=$template, Seed=$seed" >> $MAIN_LOG
        cmd="python train_enhanced_dqn.py --template $template --seed $seed --episodes $EPISODES --output $enhanced_dir"
        nohup $cmd > $enhanced_log 2>&1 &
        PID=$!
        echo "Started Enhanced DQN with PID: $PID" >> $MAIN_LOG
        echo $PID > ${enhanced_dir}/pid.txt
        
        # Neurosymbolic DQN
        neuro_dir="experiments/neuro_symbolic_comparison/$template/neurosymbolic_dqn/seed$seed"
        neuro_log="logs/${template}_neurosymbolic_dqn_seed${seed}_${START_TIME}.log"
        mkdir -p $neuro_dir
        
        echo "Starting Neurosymbolic DQN: Template=$template, Seed=$seed" >> $MAIN_LOG
        cmd="python train_neuro_symbolic.py --template $template --seed $seed --episodes $EPISODES --gradual-guidance --output $neuro_dir"
        nohup $cmd > $neuro_log 2>&1 &
        PID=$!
        echo "Started Neurosymbolic DQN with PID: $PID" >> $MAIN_LOG
        echo $PID > ${neuro_dir}/pid.txt
        
        # Add a short delay to prevent overwhelming the system
        sleep 2
    done
    
    echo "All runs started for template: $template" >> $MAIN_LOG
done

echo "All experiment runs scheduled! Check logs directory for progress."
echo "To generate plots after completion, run: python plot_neuro_symbolic_comparison.py"
