#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
from scipy.ndimage import gaussian_filter1d

def smooth(data, sigma=5):
    """Apply Gaussian smoothing to data"""
    return gaussian_filter1d(data, sigma=sigma)

def calculate_moving_average(values, window=100):
    """Calculate moving average with the specified window"""
    if len(values) < window:
        return values  # Return original if not enough data
    
    result = np.convolve(values, np.ones(window)/window, mode='valid')
    # Pad the beginning to maintain original length
    padding = np.full(window-1, np.nan)
    return np.concatenate([padding, result])

def find_training_logs(base_dir, template, config):
    """Find all training log files for a given template and configuration"""
    pattern = os.path.join(base_dir, f"template_{template}", config, "seed*", "training_log.csv")
    files = glob.glob(pattern)
    
    # Extract seed from directory path
    result = {}
    for file_path in files:
        # Extract seed from directory name
        dir_name = os.path.basename(os.path.dirname(file_path))
        match = re.search(r'seed(\d+)', dir_name)
        if match:
            seed = int(match.group(1))
            result[seed] = file_path
    
    return result

def extract_and_process_metrics(experiment_dir, template, config, seeds):
    """Extract and process metrics for a specific configuration across seeds"""
    metrics = {
        'scores': [],
        'success_rate': [],
        'wrong_key_rate': [],
        'crash_rate': [], 
        'episode_length': []
    }
    
    # Find all training log files
    log_files = find_training_logs(experiment_dir, template, config)
    
    for seed in seeds:
        if seed in log_files:
            log_file = log_files[seed]
            print(f"Processing file: {log_file}")
            
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
                    smoothed_steps = smooth(steps, sigma=15)
                    metrics['episode_length'].append((episodes, smoothed_steps))
                
            except Exception as e:
                print(f"Error processing {log_file}: {e}")
        else:
            print(f"No log file found for seed {seed} in {template}/{config}")
    
    return metrics

def generate_plots():
    """Generate plots for all metrics across templates"""
    experiment_dir = 'experiments/neuro_symbolic'
    output_dir = 'plots/neuro_symbolic'
    os.makedirs(output_dir, exist_ok=True)
    
    templates = ["basic_med", "sparse_med", "zipper_med", "bottleneck_med", "bottleneck_hard", "corridors_med"]
    configs = ["enhanced_dqn", "neurosymbolic_dqn"]
    config_labels = {
        "enhanced_dqn": "Enhanced DQN",
        "neurosymbolic_dqn": "Neurosymbolic DQN"
    }
    seeds = [42, 101, 202, 303, 404]
    
    metrics_to_plot = [
        ('scores', 'Smoothed Score', 'Score'),
        ('success_rate', 'Success Rate', 'Success Rate'),
        ('wrong_key_rate', 'Wrong Key Rate', 'Wrong Keys per Step'),
        ('crash_rate', 'Crash Rate', 'Crash Rate'),
        ('episode_length', 'Episode Length', 'Steps per Episode')
    ]
    
    # Colors for different configurations
    colors = ['#1f77b4', '#ff7f0e']
    
    # Process each template
    for template in templates:
        template_output_dir = os.path.join(output_dir, template)
        os.makedirs(template_output_dir, exist_ok=True)
        
        print(f"\nProcessing template: {template}")
        
        # Extract metrics for each configuration
        all_config_metrics = {}
        for config in configs:
            print(f"  Processing configuration: {config}")
            all_config_metrics[config] = extract_and_process_metrics(
                experiment_dir, template, config, seeds)
        
        # Generate plots for each metric
        for metric_name, title, y_label in metrics_to_plot:
            print(f"  Generating {metric_name} plot...")
            
            plt.figure(figsize=(12, 8))
            
            has_data = False
            for i, config in enumerate(configs):
                metrics = all_config_metrics[config][metric_name]
                
                if not metrics:
                    print(f"    No {metric_name} data for {config}")
                    continue
                
                # Calculate mean and standard deviation across seeds
                max_len = max(len(eps) for eps, _ in metrics) if metrics else 0
                if max_len == 0:
                    continue
                
                # Create a common x-axis (episodes)
                episode_range = np.arange(1, max_len + 1)
                
                # Align and collect values for all seeds
                values_array = []
                for episodes, values in metrics:
                    # Only use data up to the maximum common length
                    if len(episodes) > 0 and len(values) > 0:
                        # Ensure we're using the same range for all seeds
                        aligned_values = np.full(max_len, np.nan)
                        aligned_values[:len(values)] = values[:max_len]
                        values_array.append(aligned_values)
                
                if not values_array:
                    continue
                
                values_array = np.array(values_array)
                
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
                
                has_data = True
            
            if has_data:
                plt.title(f"{title} - {template.replace('_', ' ').title()}")
                plt.xlabel("Episodes")
                plt.ylabel(y_label)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(loc='best')
                
                # Save plot
                plt.savefig(os.path.join(template_output_dir, f"{metric_name}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
    # Generate overall comparison plot for success rates
    generate_overall_comparison(templates, configs, config_labels, output_dir)
    
    print(f"\nAll plots generated in {output_dir}")

def generate_overall_comparison(templates, configs, config_labels, output_dir):
    """Generate an overall comparison bar chart for final success rates"""
    experiment_dir = 'experiments/neuro_symbolic'
    seeds = [42, 101, 202, 303, 404]
    
    # Collect final success rates
    template_success = {}
    
    for template in templates:
        template_success[template] = {}
        
        for config in configs:
            metrics = extract_and_process_metrics(experiment_dir, template, config, seeds)
            success_rates = metrics['success_rate']
            
            # Calculate final average success rate (last 100 episodes)
            final_rates = []
            for episodes, rates in success_rates:
                if len(rates) >= 100:
                    final_rates.append(np.mean(rates[-100:]))
            
            if final_rates:
                template_success[template][config] = np.mean(final_rates)
            else:
                template_success[template][config] = 0
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(templates))
    width = 0.35
    
    for i, config in enumerate(configs):
        success_values = [template_success[t][config] for t in templates]
        bars = ax.bar(x + (i - 0.5) * width, success_values, width, label=config_labels[config])
        
        # Add value labels
        for bar, value in zip(bars, success_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_title('Final Success Rate by Template (Last 100 Episodes)')
    ax.set_xlabel('Template')
    ax.set_ylabel('Success Rate')
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('_', ' ').title() for t in templates])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_success_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_plots()
