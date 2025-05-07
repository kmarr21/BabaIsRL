#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import argparse
from scipy.ndimage import gaussian_filter1d

# apply gaussian smoothing to data
def smooth(data, sigma=5):
    return gaussian_filter1d(data, sigma=sigma)

# calculate moving avg with specified window
def calculate_moving_average(values, window=100):
    if len(values) < window:
        return values # return original if not enough data
    
    result = np.convolve(values, np.ones(window)/window, mode='valid')
    # pad the beginning to maintain original length
    padding = np.full(window-1, np.nan)
    return np.concatenate([padding, result])

# find all training log files for a given template & config
def find_training_logs(base_dir, template, config):
    pattern = os.path.join(base_dir, f"template_{template}", config, f"*{config}*", "training_log.csv")
    files = glob.glob(pattern)
    
    if not files:
        # try alternative pattern if first one doesn't work
        pattern = os.path.join(base_dir, f"template_{template}", config, "*", "training_log.csv")
        files = glob.glob(pattern)
    
    # extract seed from directory path
    result = {}
    for file_path in files:
        # extract seed from directory name
        dir_name = os.path.basename(os.path.dirname(file_path))
        match = re.search(r'seed(\d+)', dir_name)
        if match:
            seed = int(match.group(1))
            result[seed] = file_path
    
    return result

# extract & process metrics for specific config across seeds
def extract_and_process_metrics(experiment_dir, template, config, seeds):
    metrics = {
        'scores': [],
        'success_rate': [],
        'wrong_key_rate': [],
        'crash_rate': [], 
        'episode_length': []
    }
    
    # find all training log files
    log_files = find_training_logs(experiment_dir, template, config)
    
    for seed in seeds:
        if seed in log_files:
            log_file = log_files[seed]
            print(f"Processing file: {log_file}")
            
            try:
                df = pd.read_csv(log_file)
                
                # extract metrics
                episodes = df['episode'].values
                
                # scores (already in the dataframe)
                scores = df['score'].values
                smoothed_scores = smooth(scores)
                metrics['scores'].append((episodes, smoothed_scores))
                
                # success rate (calculate moving average)
                if 'success' in df.columns:
                    success = df['success'].values.astype(float)
                    success_rate = calculate_moving_average(success)
                    metrics['success_rate'].append((episodes, success_rate))
                
                # wrong key rate
                if 'wrong_key_attempts' in df.columns and 'steps' in df.columns:
                    # calculate wrong key rate per step
                    wrong_keys = df['wrong_key_attempts'].values
                    steps = df['steps'].values
                    # avoid division by zero
                    wrong_key_rate = np.zeros_like(wrong_keys, dtype=float)
                    mask = steps > 0
                    wrong_key_rate[mask] = wrong_keys[mask] / steps[mask]
                    # smooth rate
                    smoothed_wrong_key_rate = smooth(wrong_key_rate)
                    metrics['wrong_key_rate'].append((episodes, smoothed_wrong_key_rate))
                
                # crash rate (if termination reason is available)
                if 'termination_reason' in df.columns:
                    # calculate crash rate (moving average of terminations due to enemy_collision)
                    crashes = (df['termination_reason'] == 'enemy_collision').astype(float)
                    crash_rate = calculate_moving_average(crashes)
                    metrics['crash_rate'].append((episodes, crash_rate))
                
                # episode length
                if 'steps' in df.columns:
                    steps = df['steps'].values
                    smoothed_steps = smooth(steps, sigma=15)
                    metrics['episode_length'].append((episodes, smoothed_steps))
                
            except Exception as e:
                print(f"Error processing {log_file}: {e}")
        else:
            print(f"No log file found for seed {seed} in {template}/{config}")
    
    return metrics

# generate plots for all metrics across templates
def generate_plots(experiment_name, templates, configs, config_labels, seeds):
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
    
    # colors for different configurations
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    #process each template
    for template in templates:
        template_output_dir = os.path.join(output_dir, template)
        os.makedirs(template_output_dir, exist_ok=True)
        
        print(f"\nProcessing template: {template}")
        
        # extract metrics for each configuration
        all_config_metrics = {}
        for config in configs:
            print(f"  Processing configuration: {config}")
            all_config_metrics[config] = extract_and_process_metrics(
                experiment_dir, template, config, seeds)
        
        # generate plots for each metric
        for metric_name, title, y_label in metrics_to_plot:
            print(f"  Generating {metric_name} plot...")
            
            plt.figure(figsize=(12, 8))
            
            has_data = False
            for i, config in enumerate(configs):
                metrics = all_config_metrics[config][metric_name]
                
                if not metrics:
                    print(f"    No {metric_name} data for {config}")
                    continue
                
                # calculate mean and std dev across seeds
                max_len = max(len(eps) for eps, _ in metrics) if metrics else 0
                if max_len == 0:
                    continue
                
                # create a common x-axis (episodes)
                episode_range = np.arange(1, max_len + 1)
                
                # align and collect values for all seeds
                values_array = []
                for episodes, values in metrics:
                    # only use data up to the maximum common length
                    if len(episodes) > 0 and len(values) > 0:
                        # ensure using the same range for all seeds
                        aligned_values = np.full(max_len, np.nan)
                        aligned_values[:len(values)] = values[:max_len]
                        values_array.append(aligned_values)
                
                if not values_array:
                    continue
                
                values_array = np.array(values_array)
                
                # calculate stats, ignoring NaN values
                mean_values = np.nanmean(values_array, axis=0)
                std_values = np.nanstd(values_array, axis=0)
                
                # plot mean line
                plt.plot(episode_range, mean_values, 
                        color=colors[i % len(colors)], 
                        label=config_labels[config], 
                        linewidth=2)
                
                # plot shaded region for std dev
                plt.fill_between(episode_range, 
                                mean_values - std_values, 
                                mean_values + std_values, 
                                color=colors[i % len(colors)], 
                                alpha=0.3)
                
                has_data = True
            
            if has_data:
                plt.title(f"{title} - {template.replace('_', ' ').title()}")
                plt.xlabel("Episodes")
                plt.ylabel(y_label)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(loc='best')
                
                # save plot
                plt.savefig(os.path.join(template_output_dir, f"{metric_name}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
    print(f"\nAll plots for {experiment_name} generated in {output_dir}")

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
