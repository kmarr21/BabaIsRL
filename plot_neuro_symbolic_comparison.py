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

def calculate_moving_average(values, window=100):
    """Calculate moving average with the specified window"""
    if len(values) < window:
        return values  # Return original if not enough data
    
    result = np.convolve(values, np.ones(window)/window, mode='valid')
    # Pad the beginning to maintain original length
    padding = np.full(window-1, np.nan)
    return np.concatenate([padding, result])

def find_training_logs(base_dir, template, model_type):
    """Find all training log files for a given template and model type"""
    pattern = os.path.join(base_dir, template, model_type, "seed*", "training_log.csv")
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

def extract_and_process_metrics(experiment_dir, template, model_type, seeds):
    """Extract and process metrics for a specific model across seeds"""
    metrics = {
        'scores': [],
        'success_rate': [],
        'wrong_key_rate': [],
        'crash_rate': [], 
        'episode_length': []
    }
    
    # Additional metrics specific to neurosymbolic model
    if model_type == "neurosymbolic_dqn":
        metrics['guidance_weight'] = []
        metrics['neural_decision_rate'] = []
        metrics['guided_decision_rate'] = []
    
    # Find all training log files
    log_files = find_training_logs(experiment_dir, template, model_type)
    
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
                
                # Neurosymbolic-specific metrics
                if model_type == "neurosymbolic_dqn":
                    if 'guidance_weight' in df.columns:
                        guidance_weight = df['guidance_weight'].values
                        metrics['guidance_weight'].append((episodes, guidance_weight))
                    
                    if 'neural_decision_rate' in df.columns and 'guided_decision_rate' in df.columns:
                        neural_rate = df['neural_decision_rate'].values
                        guided_rate = df['guided_decision_rate'].values
                        metrics['neural_decision_rate'].append((episodes, neural_rate))
                        metrics['guided_decision_rate'].append((episodes, guided_rate))
                
            except Exception as e:
                print(f"Error processing {log_file}: {e}")
        else:
            print(f"No log file found for seed {seed} in {template}/{model_type}")
    
    return metrics

def generate_comparison_plots(experiment_dir, output_dir, templates, model_types, seeds):
    """Generate comparison plots for the experiment."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrics to plot
    metrics_to_plot = [
        ('scores', 'Smoothed Score', 'Score'),
        ('success_rate', 'Success Rate', 'Success Rate'),
        ('wrong_key_rate', 'Wrong Key Rate', 'Wrong Keys per Step'),
        ('crash_rate', 'Crash Rate', 'Crash Rate'),
        ('episode_length', 'Episode Length', 'Steps per Episode')
    ]
    
    # Model colors and labels
    model_colors = {
        'enhanced_dqn': '#1f77b4',  # Blue
        'neurosymbolic_dqn': '#ff7f0e'  # Orange
    }
    
    model_labels = {
        'enhanced_dqn': 'Enhanced DQN',
        'neurosymbolic_dqn': 'Neurosymbolic DQN'
    }
    
    # Process each template
    for template in templates:
        template_output_dir = os.path.join(output_dir, template)
        os.makedirs(template_output_dir, exist_ok=True)
        
        print(f"\nProcessing template: {template}")
        
        # Extract metrics for each model type
        all_model_metrics = {}
        for model_type in model_types:
            print(f"  Processing model: {model_type}")
            all_model_metrics[model_type] = extract_and_process_metrics(
                experiment_dir, template, model_type, seeds)
        
        # Generate standard comparison plots
        for metric_name, title, y_label in metrics_to_plot:
            print(f"  Generating {metric_name} comparison plot...")
            
            plt.figure(figsize=(12, 8))
            
            for model_type in model_types:
                metrics = all_model_metrics[model_type][metric_name]
                
                if not metrics:
                    print(f"    No {metric_name} data for {model_type}")
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
                        color=model_colors[model_type], 
                        label=model_labels[model_type], 
                        linewidth=2)
                
                # Plot shaded region for standard deviation
                plt.fill_between(episode_range, 
                                mean_values - std_values, 
                                mean_values + std_values, 
                                color=model_colors[model_type], 
                                alpha=0.3)
            
            plt.title(f"{title} - {template.replace('_', ' ').title()}")
            plt.xlabel("Episodes")
            plt.ylabel(y_label)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='best')
            
            # Save plot
            plt.savefig(os.path.join(template_output_dir, f"{metric_name}_comparison.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Generate neurosymbolic-specific plots
        if 'neurosymbolic_dqn' in model_types:
            print("  Generating neurosymbolic-specific plots...")
            
            # Guidance weight plot
            guidance_metrics = all_model_metrics['neurosymbolic_dqn']['guidance_weight']
            if guidance_metrics:
                plt.figure(figsize=(12, 8))
                
                # Calculate mean and std across seeds
                max_len = max(len(eps) for eps, _ in guidance_metrics)
                episode_range = np.arange(1, max_len + 1)
                
                values_array = []
                for episodes, values in guidance_metrics:
                    aligned_values = np.full(max_len, np.nan)
                    aligned_values[:len(values)] = values[:max_len]
                    values_array.append(aligned_values)
                
                values_array = np.array(values_array)
                mean_values = np.nanmean(values_array, axis=0)
                std_values = np.nanstd(values_array, axis=0)
                
                plt.plot(episode_range, mean_values, color='#2ca02c', linewidth=2, label='Mean')
                plt.fill_between(episode_range, 
                                mean_values - std_values, 
                                mean_values + std_values, 
                                color='#2ca02c', alpha=0.3, label='Std Dev')
                
                plt.title(f"Guidance Weight - {template.replace('_', ' ').title()}")
                plt.xlabel("Episodes")
                plt.ylabel("Guidance Weight")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(loc='best')
                
                plt.savefig(os.path.join(template_output_dir, "guidance_weight.png"), dpi=300, bbox_inches='tight')
                plt.close()
            
            # Neural vs Guided decision rates
            neural_metrics = all_model_metrics['neurosymbolic_dqn']['neural_decision_rate']
            guided_metrics = all_model_metrics['neurosymbolic_dqn']['guided_decision_rate']
            
            if neural_metrics and guided_metrics:
                plt.figure(figsize=(12, 8))
                
                # Calculate mean and std across seeds for both metrics
                max_len = max(max(len(eps) for eps, _ in neural_metrics), 
                              max(len(eps) for eps, _ in guided_metrics))
                episode_range = np.arange(1, max_len + 1)
                
                # Process neural rate
                neural_array = []
                for episodes, values in neural_metrics:
                    aligned_values = np.full(max_len, np.nan)
                    aligned_values[:len(values)] = values[:max_len]
                    neural_array.append(aligned_values)
                
                neural_array = np.array(neural_array)
                neural_mean = np.nanmean(neural_array, axis=0)
                neural_std = np.nanstd(neural_array, axis=0)
                
                # Process guided rate
                guided_array = []
                for episodes, values in guided_metrics:
                    aligned_values = np.full(max_len, np.nan)
                    aligned_values[:len(values)] = values[:max_len]
                    guided_array.append(aligned_values)
                
                guided_array = np.array(guided_array)
                guided_mean = np.nanmean(guided_array, axis=0)
                guided_std = np.nanstd(guided_array, axis=0)
                
                # Plot neural rate
                plt.plot(episode_range, neural_mean, color='#d62728', linewidth=2, label='Neural')
                plt.fill_between(episode_range, 
                                neural_mean - neural_std, 
                                neural_mean + neural_std, 
                                color='#d62728', alpha=0.3)
                
                # Plot guided rate
                plt.plot(episode_range, guided_mean, color='#9467bd', linewidth=2, label='Guided')
                plt.fill_between(episode_range, 
                                guided_mean - guided_std, 
                                guided_mean + guided_std, 
                                color='#9467bd', alpha=0.3)
                
                plt.title(f"Decision Rates - {template.replace('_', ' ').title()}")
                plt.xlabel("Episodes")
                plt.ylabel("Decision Rate")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(loc='best')
                
                plt.savefig(os.path.join(template_output_dir, "decision_rates.png"), dpi=300, bbox_inches='tight')
                plt.close()
        
        # Generate combined performance plot
        plt.figure(figsize=(15, 10))
        
        # Panel 1: Success Rate Comparison
        plt.subplot(2, 2, 1)
        for model_type in model_types:
            metrics = all_model_metrics[model_type]['success_rate']
            if metrics:
                # Calculate mean across seeds
                max_len = max(len(eps) for eps, _ in metrics)
                episode_range = np.arange(1, max_len + 1)
                
                values_array = []
                for episodes, values in metrics:
                    aligned_values = np.full(max_len, np.nan)
                    aligned_values[:len(values)] = values[:max_len]
                    values_array.append(aligned_values)
                
                values_array = np.array(values_array)
                mean_values = np.nanmean(values_array, axis=0)
                
                plt.plot(episode_range, mean_values, 
                        color=model_colors[model_type], 
                        label=model_labels[model_type],
                        linewidth=2)
        
        plt.title("Success Rate")
        plt.xlabel("Episodes")
        plt.ylabel("Success Rate")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        
        # Panel 2: Score Comparison
        plt.subplot(2, 2, 2)
        for model_type in model_types:
            metrics = all_model_metrics[model_type]['scores']
            if metrics:
                # Calculate mean across seeds
                max_len = max(len(eps) for eps, _ in metrics)
                episode_range = np.arange(1, max_len + 1)
                
                values_array = []
                for episodes, values in metrics:
                    aligned_values = np.full(max_len, np.nan)
                    aligned_values[:len(values)] = values[:max_len]
                    values_array.append(aligned_values)
                
                values_array = np.array(values_array)
                mean_values = np.nanmean(values_array, axis=0)
                
                plt.plot(episode_range, mean_values, 
                        color=model_colors[model_type], 
                        label=model_labels[model_type],
                        linewidth=2)
        
        plt.title("Score")
        plt.xlabel("Episodes")
        plt.ylabel("Score")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        
        # Panel 3: Wrong Key Rate Comparison
        plt.subplot(2, 2, 3)
        for model_type in model_types:
            metrics = all_model_metrics[model_type]['wrong_key_rate']
            if metrics:
                # Calculate mean across seeds
                max_len = max(len(eps) for eps, _ in metrics)
                episode_range = np.arange(1, max_len + 1)
                
                values_array = []
                for episodes, values in metrics:
                    aligned_values = np.full(max_len, np.nan)
                    aligned_values[:len(values)] = values[:max_len]
                    values_array.append(aligned_values)
                
                values_array = np.array(values_array)
                mean_values = np.nanmean(values_array, axis=0)
                
                plt.plot(episode_range, mean_values, 
                        color=model_colors[model_type], 
                        label=model_labels[model_type],
                        linewidth=2)
        
        plt.title("Wrong Key Rate")
        plt.xlabel("Episodes")
        plt.ylabel("Wrong Keys per Step")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        
        # Panel 4: Episode Length Comparison
        plt.subplot(2, 2, 4)
        for model_type in model_types:
            metrics = all_model_metrics[model_type]['episode_length']
            if metrics:
                # Calculate mean across seeds
                max_len = max(len(eps) for eps, _ in metrics)
                episode_range = np.arange(1, max_len + 1)
                
                values_array = []
                for episodes, values in metrics:
                    aligned_values = np.full(max_len, np.nan)
                    aligned_values[:len(values)] = values[:max_len]
                    values_array.append(aligned_values)
                
                values_array = np.array(values_array)
                mean_values = np.nanmean(values_array, axis=0)
                
                plt.plot(episode_range, mean_values, 
                        color=model_colors[model_type], 
                        label=model_labels[model_type],
                        linewidth=2)
        
        plt.title("Episode Length")
        plt.xlabel("Episodes")
        plt.ylabel("Steps")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        
        plt.tight_layout()
        plt.suptitle(f"Performance Comparison - {template.replace('_', ' ').title()}", fontsize=16, y=1.02)
        plt.savefig(os.path.join(template_output_dir, "combined_performance.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate overall summary performance plot across all templates
        print(f"Processing completed for template: {template}")
    
    # Generate overall comparison plot
    generate_overall_comparison_plot(experiment_dir, output_dir, templates, model_types, seeds)
    
    print(f"\nAll plots generated in {output_dir}")

def generate_overall_comparison_plot(experiment_dir, output_dir, templates, model_types, seeds):
    """Generate overall comparison plot across all templates."""
    print("Generating overall comparison plot across all templates...")
    
    # Model colors and labels
    model_colors = {
        'enhanced_dqn': '#1f77b4',  # Blue
        'neurosymbolic_dqn': '#ff7f0e'  # Orange
    }
    
    model_labels = {
        'enhanced_dqn': 'Enhanced DQN',
        'neurosymbolic_dqn': 'Neurosymbolic DQN'
    }
    
    # Collect final success rates for each template and model
    final_success_rates = {model: {} for model in model_types}
    
    for template in templates:
        for model_type in model_types:
            metrics = extract_and_process_metrics(experiment_dir, template, model_type, seeds)
            
            # Get success rate data
            success_metrics = metrics['success_rate']
            if success_metrics:
                # Calculate average final success rate across seeds
                final_rates = []
                for episodes, values in success_metrics:
                    # Get final 100 episodes success rate
                    final_window = min(100, len(values))
                    final_rate = np.nanmean(values[-final_window:])
                    final_rates.append(final_rate)
                
                if final_rates:
                    final_success_rates[model_type][template] = np.mean(final_rates)
    
    # Create bar chart comparing final success rates
    plt.figure(figsize=(14, 8))
    
    # Get template names and organize data for plotting
    plot_templates = sorted(templates)
    width = 0.35
    x = np.arange(len(plot_templates))
    
    # Plot bars for each model
    bars = []
    for i, model_type in enumerate(model_types):
        values = [final_success_rates[model_type].get(template, 0) for template in plot_templates]
        bar = plt.bar(x + i*width - width/2, values, width, label=model_labels[model_type], color=model_colors[model_type])
        bars.append(bar)
    
    # Add value labels on top of bars
    for bar in bars:
        for rect in bar:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., height + 0.02,
                     f'{height:.2f}',
                     ha='center', va='bottom', rotation=0)
    
    plt.title("Final Success Rate by Template (Last 100 Episodes)", fontsize=16)
    plt.xlabel("Template", fontsize=14)
    plt.ylabel("Success Rate", fontsize=14)
    plt.ylim(0, 1.1)  # Set y-axis limit
    plt.xticks(x, [t.replace('_', ' ').title() for t in plot_templates], rotation=45, ha='right')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(model_types))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    plt.savefig(os.path.join(output_dir, "overall_success_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Overall comparison plot generated successfully.")

def main():
    parser = argparse.ArgumentParser(description="Generate comparative plots for neurosymbolic experiment")
    parser.add_argument("--experiment-dir", type=str, default="experiments/neuro_symbolic_comparison",
                       help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=str, default="~/neuro_symbolic_plots",
                       help="Directory to save generated plots")
    parser.add_argument("--templates", nargs="+", default=["basic_med", "sparse_med", "zipper_med", 
                                                          "bottleneck_med", "bottleneck_hard", "corridors_med"],
                       help="Templates to process")
    parser.add_argument("--models", nargs="+", default=["enhanced_dqn", "neurosymbolic_dqn"],
                       help="Model types to compare")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 202, 303, 404],
                       help="Seeds to include in the analysis")
    
    args = parser.parse_args()
    
    # Expand the output directory path if needed
    output_dir = os.path.expanduser(args.output_dir)
    
    print("Starting neurosymbolic comparison plot generation")
    print(f"Experiment directory: {args.experiment_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Templates: {args.templates}")
    print(f"Models: {args.models}")
    print(f"Seeds: {args.seeds}")
    
    # Generate all plots
    generate_comparison_plots(args.experiment_dir, output_dir, args.templates, args.models, args.seeds)

if __name__ == "__main__":
    main()
