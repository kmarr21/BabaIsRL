#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import argparse
from scipy.ndimage import gaussian_filter1d

def smooth(data, sigma=5):
    """Apply Gaussian smoothing to data"""
    return gaussian_filter1d(data, sigma=sigma)

def extract_metric_from_log(log_file, metric_name):
    """Extract a metric from a CSV log file"""
    try:
        df = pd.read_csv(log_file)
        return df['episode'].values, df[metric_name].values
    except Exception as e:
        print(f"Error extracting {metric_name} from {log_file}: {e}")
        return None, None

def calculate_moving_average(values, window=100):
    """Calculate moving average with the specified window"""
    if len(values) < window:
        return values  # Return original if not enough data
    
    result = np.convolve(values, np.ones(window)/window, mode='valid')
    # Pad the beginning to maintain original length
    padding = np.full(window-1, np.nan)
    return np.concatenate([padding, result])

def extract_and_process_metrics(experiment_dir, template, config, seeds):
    """Extract and process metrics for a specific configuration across seeds"""
    metrics = {
        'scores': [],
        'success_rate': [],
        'wrong_key_rate': [],
        'crash_rate': [], 
        'episode_length': []
    }
    
    for seed in seeds:
        seed_dir = os.path.join(experiment_dir, f"template_{template}", config, f"seed{seed}")
        log_file = os.path.join(seed_dir, "training_log.csv")
        
        if not os.path.exists(log_file):
            print(f"Warning: Log file not found: {log_file}")
            continue
        
        try:
            df = pd.read_csv(log_file)
            
            # Extract metrics
            episodes = df['episode'].values
            
            # Scores (already in the dataframe)
            scores = df['score'].values
            smoothed_scores = smooth(scores)
            metrics['scores'].append((episodes, smoothed_scores))
            
            # Success rate (calculate moving average)
            if 'success' in df.columns:
                success = df['success'].values.astype(float)
                success_rate = calculate_moving_average(success)
                metrics['success_rate'].append((episodes, success_rate))
            
            # Wrong key rate
            if 'wrong_key_attempts' in df.columns and 'steps' in df.columns:
                # Calculate wrong key rate per step
                wrong_keys = df['wrong_key_attempts'].values
                steps = df['steps'].values
                # Avoid division by zero
                wrong_key_rate = np.zeros_like(wrong_keys, dtype=float)
                mask = steps > 0
                wrong_key_rate[mask] = wrong_keys[mask] / steps[mask]
                # Smooth the rate
                smoothed_wrong_key_rate = smooth(wrong_key_rate)
                metrics['wrong_key_rate'].append((episodes, smoothed_wrong_key_rate))
            
            # Crash rate (if termination reason is available)
            if 'termination_reason' in df.columns:
                # Calculate crash rate (moving average of terminations due to enemy_collision)
                crashes = (df['termination_reason'] == 'enemy_collision').astype(float)
                crash_rate = calculate_moving_average(crashes)
                metrics['crash_rate'].append((episodes, crash_rate))
            
            # Episode length
            if 'steps' in df.columns:
                steps = df['steps'].values
                smoothed_steps = smooth(steps)
                metrics['episode_length'].append((episodes, smoothed_steps))
            
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
    
    return metrics

def generate_plots(experiment_name, templates, configs, config_labels, seeds):
    """Generate plots for all metrics across templates"""
    experiment_dir = os.path.join('experiments', experiment_name)
    output_dir = os.path.join('plots', experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_to_plot = [
        ('scores', 'Smoothed Score', 'Score'),
        ('success_rate', 'Success Rate', 'Success Rate'),
        ('wrong_key_rate', 'Wrong Key Rate', 'Wrong Keys per Step'),
        ('crash_rate', 'Crash Rate', 'Crash Rate'),
        ('episode_length', 'Episode Length', 'Steps per Episode')
    ]
    
    # Colors for different configurations
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Process each template
    for template in templates:
        template_output_dir = os.path.join(output_dir, template)
        os.makedirs(template_output_dir, exist_ok=True)
        
        print(f"Processing template: {template}")
        
        # Extract metrics for each configuration
        all_config_metrics = {}
        for config in configs:
            all_config_metrics[config] = extract_and_process_metrics(
                experiment_dir, template, config, seeds)
        
        # Generate plots for each metric
        for metric_name, title, y_label in metrics_to_plot:
            print(f"  Generating {metric_name} plot...")
            
            plt.figure(figsize=(12, 8))
            
            for i, config in enumerate(configs):
                metrics = all_config_metrics[config][metric_name]
                
                if not metrics:
                    print(f"    No {metric_name} data for {config}")
                    continue
                
                # Calculate mean and standard deviation across seeds
                max_episodes = max(episodes.max() for episodes, _ in metrics if len(episodes) > 0)
                episode_range = np.arange(1, max_episodes + 1)
                values_array = np.zeros((len(metrics), len(episode_range)))
                values_array.fill(np.nan)
                
                for j, (episodes, values) in enumerate(metrics):
                    if len(episodes) > 0:
                        # Ensure all arrays have the same length by padding with NaN
                        valid_indices = episodes - 1  # Convert to 0-based indexing
                        if len(valid_indices) == len(values):
                            values_array[j, valid_indices] = values
                
                # Calculate statistics, ignoring NaN values
                mean_values = np.nanmean(values_array, axis=0)
                std_values = np.nanstd(values_array, axis=0)
                
                # Plot mean line
                plt.plot(episode_range, mean_values, 
                         color=colors[i % len(colors)], 
                         label=config_labels[config], 
                         linewidth=2)
                
                # Plot shaded region for standard deviation
                plt.fill_between(episode_range, 
                                mean_values - std_values, 
                                mean_values + std_values, 
                                color=colors[i % len(colors)], 
                                alpha=0.3)
            
            plt.title(f"{title} - {template.replace('_', ' ').title()}")
            plt.xlabel("Episodes")
            plt.ylabel(y_label)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='best')
            
            # Save plot
            plt.savefig(os.path.join(template_output_dir, f"{metric_name}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
    print(f"All plots for {experiment_name} generated in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate plots from experiment results")
    parser.add_argument("experiments", nargs="+", help="Experiments to plot (experiment1, experiment2, or both)")
    args = parser.parse_args()
    
    templates = ["basic_med", "sparse_med", "zipper_med", "bottleneck_med", "bottleneck_hard", "corridors_med"]
    seeds = [42, 101, 202, 303, 404]
    
    for experiment in args.experiments:
        if experiment == "experiment1":
            configs = ["baseline_dqn", "reward_shaping_only", "state_aug_only", "full_enhanced_dqn"]
            config_labels = {
                "baseline_dqn": "Baseline DQN",
                "reward_shaping_only": "Reward Shaping Only",
                "state_aug_only": "State Augmentation Only",
                "full_enhanced_dqn": "Full Enhanced DQN"
            }
            generate_plots("experiment1", templates, configs, config_labels, seeds)
            
        elif experiment == "experiment2":
            configs = ["full_enhanced_dqn", "standard_ksm", "adaptive_ksm"]
            config_labels = {
                "full_enhanced_dqn": "Full Enhanced DQN",
                "standard_ksm": "Standard KSM",
                "adaptive_ksm": "Adaptive KSM"
            }
            generate_plots("experiment2", templates, configs, config_labels, seeds)

if __name__ == "__main__":
    main()
