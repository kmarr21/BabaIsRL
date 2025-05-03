# plot_neuro_symbolic_results.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import argparse
from scipy.ndimage import gaussian_filter1d

def smooth(data, sigma=5):
    """Apply Gaussian smoothing to data"""
    return gaussian_filter1d(data, sigma=sigma)

def load_training_data(log_file):
    """Load training data from a log file"""
    try:
        df = pd.read_csv(log_file)
        return df
    except Exception as e:
        print(f"Error loading data from {log_file}: {e}")
        return None

def generate_standard_plots(template, seeds, output_dir):
    """Generate standard plots for a template across multiple seeds"""
    print(f"Generating standard plots for template: {template}")
    
    # Create output directory
    plots_dir = os.path.join(output_dir, template)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load data from all seeds
    log_files = []
    for seed in seeds:
        log_file = f"neuro_symbolic_results/template_{template}/seed{seed}/training_log.csv"
        if os.path.exists(log_file):
            log_files.append(log_file)
        else:
            print(f"Warning: Log file not found for template {template}, seed {seed}")
    
    if not log_files:
        print(f"No log files found for template {template}")
        return
    
    # Load dataframes
    dfs = []
    for log_file in log_files:
        df = load_training_data(log_file)
        if df is not None:
            dfs.append(df)
    
    if not dfs:
        print(f"No valid dataframes loaded for template {template}")
        return
    
    # Metrics to plot
    metrics = [
        ('score', 'Smoothed Score', 'Score'),
        ('success', 'Success Rate', 'Rate'),
        ('wrong_key_attempts', 'Wrong Key Rate', 'Rate (per step)'),
        ('steps', 'Episode Length', 'Steps'),
        ('guided_decision_rate', 'Guided Decision Rate', 'Rate')
    ]
    
    # Generate plots for each metric
    for metric_name, title, y_label in metrics:
        plt.figure(figsize=(10, 6))
        
        # Initialize arrays for collecting data across seeds
        max_episodes = max(df['episode'].max() for df in dfs)
        episode_range = np.arange(1, max_episodes + 1)
        values_array = np.zeros((len(dfs), len(episode_range)))
        values_array.fill(np.nan)
        
        # Fill array with values from each seed
        for i, df in enumerate(dfs):
            episodes = df['episode'].values
            
            if metric_name == 'wrong_key_attempts' and 'steps' in df.columns:
                # Calculate wrong key rate per step
                values = df[metric_name].values / np.maximum(1, df['steps'].values)
            else:
                values = df[metric_name].values if metric_name in df.columns else np.zeros_like(episodes)
            
            # Apply smoothing
            smoothed_values = smooth(values)
            
            # Insert values at correct episode indices (convert to 0-based)
            valid_indices = episodes - 1
            if len(valid_indices) == len(smoothed_values):
                values_array[i, valid_indices] = smoothed_values
        
        # Calculate mean and standard deviation
        mean_values = np.nanmean(values_array, axis=0)
        std_values = np.nanstd(values_array, axis=0)
        
        # Clip standard deviation for bounded metrics
        if metric_name == 'success' or metric_name == 'guided_decision_rate':
            upper_bound = np.minimum(mean_values + std_values, 1.0)
            lower_bound = np.maximum(mean_values - std_values, 0.0)
        else:
            upper_bound = mean_values + std_values
            lower_bound = mean_values - std_values
        
        # Plot mean line
        plt.plot(episode_range, mean_values, color='#2ca02c', linewidth=2, label='Mean')
        
        # Plot shaded region for standard deviation
        plt.fill_between(episode_range, lower_bound, upper_bound, color='#2ca02c', alpha=0.3, label='Std Dev')
        
        plt.title(f"{title} - {template.replace('_', ' ').title()} Template")
        plt.xlabel("Episodes")
        plt.ylabel(y_label)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(plots_dir, f"{metric_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots generated for template {template}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate standard plots for Neurosymbolic DQN results")
    parser.add_argument("--templates", nargs="+", default=["basic_med", "sparse_med", "zipper_med", 
                                                         "bottleneck_med", "bottleneck_hard", "corridors_med"],
                       help="Templates to generate plots for")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 202],
                       help="Seeds to include in the plots")
    parser.add_argument("--output", type=str, default="neuro_symbolic_plots",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate plots for each template
    for template in args.templates:
        generate_standard_plots(template, args.seeds, args.output)
    
    print("All plots generated!")
