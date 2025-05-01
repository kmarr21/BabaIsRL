import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d

def smooth_data(data, sigma=5):
    """Apply Gaussian smoothing to data."""
    return gaussian_filter1d(data, sigma=sigma)

def plot_with_std(ax, x, y_mean, y_std, label, color=None, alpha=0.3, sigma=5):
    """Plot a line with shaded region for standard deviation."""
    if sigma > 0:
        y_mean = smooth_data(y_mean, sigma)
        
    line = ax.plot(x, y_mean, label=label, color=color)
    color = line[0].get_color() if color is None else color
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=alpha)
    return line[0].get_color()

def create_comparison_plots(results, output_dir, template_name, num_episodes, experiment_name):
    """
    Create comparative plots for different models.
    
    Args:
        results: Dictionary where keys are model names and values are dictionaries of metrics
                Each metric contains arrays for mean and std across seeds
        output_dir: Directory to save plots
        template_name: Name of the environment template used
        num_episodes: Number of episodes in the experiment
        experiment_name: Name of the experiment for plot titles
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['smoothed_scores', 'success_rates', 'episode_lengths', 'wrong_key_rates', 'crash_counts']
    titles = ['Smoothed Scores', 'Success Rate', 'Episode Length', 'Wrong Key Attempt Rate', 'Total Robot Crashes']
    ylabels = ['Score', 'Success Rate', 'Steps', 'Wrong Key Rate', 'Total Crashes']
    
    x = np.arange(num_episodes)
    
    # Create a single figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Set a common style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Plot each metric
    for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        if i < len(axes):
            ax = axes[i]
            
            for model_name in results.keys():
                if metric in results[model_name]:
                    mean_data = results[model_name][metric]['mean']
                    std_data = results[model_name][metric]['std']
                    
                    # Ensure data length matches x
                    if len(mean_data) < len(x):
                        padded_mean = np.pad(mean_data, (0, len(x) - len(mean_data)), 'edge')
                        padded_std = np.pad(std_data, (0, len(x) - len(std_data)), 'edge')
                        plot_with_std(ax, x, padded_mean, padded_std, model_name)
                    else:
                        plot_with_std(ax, x, mean_data[:len(x)], std_data[:len(x)], model_name)
            
            ax.set_title(f'{title} - {template_name}')
            ax.set_xlabel('Episode')
            ax.set_ylabel(ylabel)
            ax.legend(loc='best')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
    # Remove any unused subplots
    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{experiment_name}_{template_name}_combined.png", dpi=300)
    plt.close()
    
    # Now create individual plots for each metric (larger, more detailed)
    for metric, title, ylabel in zip(metrics, titles, ylabels):
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        
        for model_name in results.keys():
            if metric in results[model_name]:
                mean_data = results[model_name][metric]['mean']
                std_data = results[model_name][metric]['std']
                
                # Ensure data length matches x
                if len(mean_data) < len(x):
                    padded_mean = np.pad(mean_data, (0, len(x) - len(mean_data)), 'edge')
                    padded_std = np.pad(std_data, (0, len(x) - len(std_data)), 'edge')
                    plot_with_std(ax, x, padded_mean, padded_std, model_name)
                else:
                    plot_with_std(ax, x, mean_data[:len(x)], std_data[:len(x)], model_name)
        
        plt.title(f'{title} - {template_name}')
        plt.xlabel('Episode')
        plt.ylabel(ylabel)
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{experiment_name}_{template_name}_{metric}.png", dpi=300)
        plt.close()
        
def load_and_process_results(base_dir, model_configs, num_seeds=5, window_size=100):
    """
    Load and process results from multiple runs.
    
    Args:
        base_dir: Base directory for results
        model_configs: List of dictionaries, each with keys 'name', 'augmented', 'shaped', 'ksm_mode'
        num_seeds: Number of seeds per model
        window_size: Window size for moving average calculations
    
    Returns:
        Dictionary of processed results for each model
    """
    results = {}
    
    for config in model_configs:
        model_name = config['name']
        results[model_name] = {}
        
        # Prepare arrays for each metric across seeds
        scores_by_seed = []
        success_rates_by_seed = []
        episode_lengths_by_seed = []
        wrong_key_rates_by_seed = []
        crash_counts_by_seed = []
        
        for seed in range(num_seeds):
            aug_str = "augmented" if config['augmented'] else "basic"
            shaping_str = "shaped" if config['shaped'] else "raw"
            ksm_str = config['ksm_mode'] if config['ksm_mode'] != "off" else "no_ksm"
            
            # Directory pattern based on configuration
            result_dir = f"{base_dir}_{config['template']}_{aug_str}_{shaping_str}_{ksm_str}_seed{seed}"
            
            # Load log data
            log_file = os.path.join(result_dir, 'training_log.csv')
            if not os.path.exists(log_file):
                print(f"Warning: Log file not found at {log_file}")
                continue
                
            log_data = pd.read_csv(log_file)
            
            # Extract metrics
            scores = log_data['score'].values
            steps = log_data['steps'].values
            success = log_data['success'].values
            wrong_key_attempts = log_data['wrong_key_attempts'].values
            
            # Calculate metrics
            # Smoothed scores (moving average)
            smoothed_scores = np.convolve(scores, np.ones(min(window_size, len(scores)))/min(window_size, len(scores)), mode='valid')
            
            # Success rates (moving window)
            success_rates = []
            for i in range(len(success)):
                if i < window_size:
                    success_rates.append(np.mean(success[:i+1]))
                else:
                    success_rates.append(np.mean(success[i-window_size+1:i+1]))
            
            # Episode lengths (moving average)
            episode_lengths = []
            for i in range(len(steps)):
                if i < window_size:
                    episode_lengths.append(np.mean(steps[:i+1]))
                else:
                    episode_lengths.append(np.mean(steps[i-window_size+1:i+1]))
            
            # Wrong key attempt rates
            wrong_key_rates = []
            for i in range(len(wrong_key_attempts)):
                if i < window_size:
                    total_steps = np.sum(steps[:i+1])
                    if total_steps > 0:
                        wrong_key_rates.append(np.sum(wrong_key_attempts[:i+1]) / total_steps)
                    else:
                        wrong_key_rates.append(0)
                else:
                    total_steps = np.sum(steps[i-window_size+1:i+1])
                    if total_steps > 0:
                        wrong_key_rates.append(np.sum(wrong_key_attempts[i-window_size+1:i+1]) / total_steps)
                    else:
                        wrong_key_rates.append(0)
            
            # Crash counts (cumulative)
            termination_reasons = log_data['termination_reason'].values
            crash_count = 0
            crash_counts = []
            for reason in termination_reasons:
                if reason == 'enemy_collision':
                    crash_count += 1
                crash_counts.append(crash_count)
            
            # Append results from this seed
            scores_by_seed.append(smoothed_scores)
            success_rates_by_seed.append(success_rates)
            episode_lengths_by_seed.append(episode_lengths)
            wrong_key_rates_by_seed.append(wrong_key_rates)
            crash_counts_by_seed.append(crash_counts)
        
        # Calculate mean and std for each metric across seeds
        if scores_by_seed:
            # Find the minimum length across all seeds for each metric
            min_score_len = min(len(s) for s in scores_by_seed)
            min_success_len = min(len(s) for s in success_rates_by_seed)
            min_length_len = min(len(s) for s in episode_lengths_by_seed)
            min_wrong_key_len = min(len(s) for s in wrong_key_rates_by_seed)
            min_crash_len = min(len(s) for s in crash_counts_by_seed)
            
            # Truncate to minimum length and convert to numpy arrays
            scores_array = np.array([s[:min_score_len] for s in scores_by_seed])
            success_array = np.array([s[:min_success_len] for s in success_rates_by_seed])
            length_array = np.array([s[:min_length_len] for s in episode_lengths_by_seed])
            wrong_key_array = np.array([s[:min_wrong_key_len] for s in wrong_key_rates_by_seed])
            crash_array = np.array([s[:min_crash_len] for s in crash_counts_by_seed])
            
            # Calculate mean and std
            results[model_name]['smoothed_scores'] = {
                'mean': np.mean(scores_array, axis=0),
                'std': np.std(scores_array, axis=0)
            }
            results[model_name]['success_rates'] = {
                'mean': np.mean(success_array, axis=0),
                'std': np.std(success_array, axis=0)
            }
            results[model_name]['episode_lengths'] = {
                'mean': np.mean(length_array, axis=0),
                'std': np.std(length_array, axis=0)
            }
            results[model_name]['wrong_key_rates'] = {
                'mean': np.mean(wrong_key_array, axis=0),
                'std': np.std(wrong_key_array, axis=0)
            }
            results[model_name]['crash_counts'] = {
                'mean': np.mean(crash_array, axis=0),
                'std': np.std(crash_array, axis=0)
            }
        
    return results
