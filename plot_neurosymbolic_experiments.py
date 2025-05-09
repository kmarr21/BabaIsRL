#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
from scipy.ndimage import gaussian_filter1d

# apply gaussian smoothing to data
def smooth(data, sigma=5):
    return gaussian_filter1d(data, sigma=sigma)

# calculate moving avg with specified window
def calculate_moving_average(values, window=100):
    if len(values) < window:
        return values  # return original if not enough data
    
    result = np.convolve(values, np.ones(window)/window, mode='valid')
    # pad the beginning to maintain original length
    padding = np.full(window-1, np.nan)
    return np.concatenate([padding, result])

# find all training log files for given template & config
def find_training_logs(base_dir, template, config, seeds):
    result = {}
    
    for seed in seeds:
        # try multiple potential patterns
        patterns = [
            os.path.join(base_dir, f"template_{template}", config, f"seed{seed}", "training_log.csv"),
            os.path.join(base_dir, f"template_{template}", config, f"seed{seed}_*", "training_log.csv"),
            os.path.join(base_dir, f"template_{template}", config, f"*seed{seed}*", "training_log.csv")
        ]
        
        found = False
        for pattern in patterns:
            files = glob.glob(pattern)
            if files:
                result[seed] = files[0]
                found = True
                break
        
        if not found:
            print(f"No log file found for seed {seed} in {template}/{config}")
    
    return result

# process a template and generate plots
def process_template(experiment_dir, template, configs, config_labels, seeds, output_dir):
    print(f"Processing template: {template}")
    
    # metrics to plot
    metrics_to_plot = [
        ('scores', 'Smoothed Score', 'Score'),
        ('success_rate', 'Success Rate', 'Success Rate'),
        ('wrong_key_rate', 'Wrong Key Rate', 'Wrong Keys per Step'),
        ('crash_rate', 'Crash Rate', 'Crash Rate'),
        ('episode_length', 'Episode Length', 'Steps per Episode')
    ]
    
    # colors for different configurations
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # extract metrics for each configuration
    all_config_metrics = {}
    for config in configs:
        print(f"  Looking for data for configuration: {config}")
        all_config_metrics[config] = {}
        
        # find all training log files
        log_files = find_training_logs(experiment_dir, template, config, seeds)
        
        if not log_files:
            print(f"  No log files found for {template}/{config}")
            continue
            
        # print found files for debugging
        print(f"  Found log files for {config}: {len(log_files)}")
        for seed, file in log_files.items():
            print(f"    Seed {seed}: {file}")
        
        # process each log file
        for seed, log_file in log_files.items():
            print(f"  Processing file: {log_file}")
            
            try:
                df = pd.read_csv(log_file)
                episodes = df['episode'].values
                
                # initialize metrics if not already done
                for metric_name, _, _ in metrics_to_plot:
                    if metric_name not in all_config_metrics[config]:
                        all_config_metrics[config][metric_name] = []
                
                # extract and process metrics
                # scores
                scores = df['score'].values
                smoothed_scores = smooth(scores)
                all_config_metrics[config]['scores'].append((episodes, smoothed_scores))
                
                # success rate
                if 'success' in df.columns:
                    success = df['success'].values.astype(float)
                    success_rate = calculate_moving_average(success)
                    all_config_metrics[config]['success_rate'].append((episodes, success_rate))
                
                # wrong key rate
                if 'wrong_key_attempts' in df.columns and 'steps' in df.columns:
                    wrong_keys = df['wrong_key_attempts'].values
                    steps = df['steps'].values
                    wrong_key_rate = np.zeros_like(wrong_keys, dtype=float)
                    mask = steps > 0
                    wrong_key_rate[mask] = wrong_keys[mask] / steps[mask]
                    smoothed_wrong_key_rate = smooth(wrong_key_rate)
                    all_config_metrics[config]['wrong_key_rate'].append((episodes, smoothed_wrong_key_rate))
                
                # crash rate
                if 'termination_reason' in df.columns:
                    crashes = (df['termination_reason'] == 'enemy_collision').astype(float)
                    crash_rate = calculate_moving_average(crashes)
                    all_config_metrics[config]['crash_rate'].append((episodes, crash_rate))
                
                # episode length
                if 'steps' in df.columns:
                    steps = df['steps'].values
                    smoothed_steps = smooth(steps, sigma=15)
                    all_config_metrics[config]['episode_length'].append((episodes, smoothed_steps))
                
            except Exception as e:
                print(f"Error processing {log_file}: {e}")
    
    # generate plots for each metric
    for metric_name, title, y_label in metrics_to_plot:
        print(f"  Generating {metric_name} plot...")
        
        plt.figure(figsize=(12, 8))
        
        has_data = False
        for i, config in enumerate(configs):
            if config not in all_config_metrics:
                print(f"    No data for {config}")
                continue
                
            metrics = all_config_metrics[config].get(metric_name, [])
            
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
                # Only use data up to the maximum common length
                if len(episodes) > 0 and len(values) > 0:
                    # Ensure we're using the same range for all seeds
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
                    label=config_labels[i], 
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
            plt.savefig(os.path.join(output_dir, f"{metric_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots generated for template {template}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate plots for neurosymbolic experiments")
    parser.add_argument("--output-dir", default="~/neurosymbolic_plots", help="Output directory for plots")
    parser.add_argument("--data-dir", default="~/BabaIsRL/experiments", help="Directory containing experiment data")
    args = parser.parse_args()
    
    # expand paths
    output_dir = os.path.expanduser(args.output_dir)
    data_dir = os.path.expanduser(args.data_dir)
    
    # add timestamp to output directory
    timestamp = os.path.basename(output_dir)
    if "plots" in timestamp and "_" not in timestamp:
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}_{timestamp}"
    
    # templates and seeds
    templates = ["basic_med", "sparse_med", "zipper_med", "bottleneck_med", "bottleneck_hard", "corridors_med"]
    seeds = [42, 101, 202, 303, 404]
    
    # process neuro_symbolic
    experiment = "neuro_symbolic"
    experiment_dir = os.path.join(data_dir, experiment)
    experiment_output_dir = os.path.join(output_dir, experiment)
    os.makedirs(experiment_output_dir, exist_ok=True)
    
    configs = ["enhanced_dqn", "neurosymbolic_dqn"]
    config_labels = ["Enhanced DQN", "Neurosymbolic DQN"]
    
    for template in templates:
        template_output_dir = os.path.join(experiment_output_dir, template)
        os.makedirs(template_output_dir, exist_ok=True)
        process_template(experiment_dir, template, configs, config_labels, seeds, template_output_dir)
    
    # process neuro_symbolic_comparison if it exists
    comparison_experiment = "neuro_symbolic_comparison"
    comparison_dir = os.path.join(data_dir, comparison_experiment)
    if os.path.exists(comparison_dir):
        experiment_output_dir = os.path.join(output_dir, comparison_experiment)
        os.makedirs(experiment_output_dir, exist_ok=True)
        
        for template in templates:
            template_output_dir = os.path.join(experiment_output_dir, template)
            os.makedirs(template_output_dir, exist_ok=True)
            process_template(comparison_dir, template, configs, config_labels, seeds, template_output_dir)
    
    # create zip archives for each template
    for experiment in ["neuro_symbolic", "neuro_symbolic_comparison"]:
        experiment_dir = os.path.join(data_dir, experiment)
        if not os.path.exists(experiment_dir):
            continue
            
        experiment_output_dir = os.path.join(output_dir, experiment)
        for template in templates:
            template_output_dir = os.path.join(experiment_output_dir, template)
            if os.path.exists(template_output_dir):
                import zipfile
                zip_file = os.path.join(output_dir, f"{experiment}_{template}.zip")
                with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(template_output_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, experiment_output_dir)
                            zipf.write(file_path, arcname)
    
    # also collect all CSV files
    csv_dir = os.path.join(output_dir, "csv_files")
    os.makedirs(csv_dir, exist_ok=True)
    
    for experiment in ["neuro_symbolic", "neuro_symbolic_comparison"]:
        experiment_dir = os.path.join(data_dir, experiment)
        if not os.path.exists(experiment_dir):
            continue
            
        experiment_csv_dir = os.path.join(csv_dir, experiment)
        os.makedirs(experiment_csv_dir, exist_ok=True)
        
        for template in templates:
            for config in configs:
                config_csv_dir = os.path.join(experiment_csv_dir, template, config)
                os.makedirs(config_csv_dir, exist_ok=True)
                
                for seed in seeds:
                    # find CSV
                    log_files = find_training_logs(experiment_dir, template, config, [seed])
                    if seed in log_files:
                        log_file = log_files[seed]
                        dest_file = os.path.join(config_csv_dir, f"seed{seed}_training_log.csv")
                        # Copy file
                        import shutil
                        shutil.copy2(log_file, dest_file)
                        print(f"Copied {log_file} to {dest_file}")
    
    # create zip of CSV files
    import zipfile
    csv_zip = os.path.join(output_dir, "neurosymbolic_csv_files.zip")
    with zipfile.ZipFile(csv_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(csv_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, csv_dir)
                zipf.write(file_path, arcname)
    
    print(f"\nAll plots generated in: {output_dir}")
    print(f"Zip archives are available in the same directory")
    print(f"CSV files collected in: {csv_zip}")

if __name__ == "__main__":
    main()
