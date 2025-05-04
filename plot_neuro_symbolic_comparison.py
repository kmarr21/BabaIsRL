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

def extract_metrics(log_files):
    """Extract metrics from log files"""
    # Initialize arrays for storing metrics
    all_success_rates = []
    all_scores = []
    all_episodes = []
    
    # Process each log file
    for seed, log_file in log_files.items():
        try:
            df = pd.read_csv(log_file)
            episodes = df['episode'].values
            
            # Extract metrics
            if 'success' in df.columns:
                # Calculate rolling success rate
                window = 100
                success = df['success'].values
                success_rate = pd.Series(success).rolling(window=window, min_periods=1).mean().values
                all_success_rates.append((episodes, success_rate))
            
            # Extract scores
            if 'score' in df.columns:
                scores = df['score'].values
                smoothed_scores = smooth(scores)
                all_scores.append((episodes, smoothed_scores))
                
            all_episodes.append(episodes)
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
    
    return all_episodes, all_success_rates, all_scores

def plot_metrics(template, enhanced_metrics, neuro_metrics, output_dir):
    """Generate comparison plots for metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot success rate
    _, enhanced_success, enhanced_scores = enhanced_metrics
    _, neuro_success, neuro_scores = neuro_metrics
    
    # Determine maximum episode number
    max_episodes = max([eps[-1] for eps, _ in enhanced_success + neuro_success] if (enhanced_success or neuro_success) else [0])
    
    # Setup figure for success rate
    plt.figure(figsize=(12, 8))
    
    # Plot success rate for enhanced DQN
    for episodes, success_rate in enhanced_success:
        plt.plot(episodes, success_rate, 'b-', alpha=0.3)
    
    # Plot success rate for neurosymbolic DQN
    for episodes, success_rate in neuro_success:
        plt.plot(episodes, success_rate, 'r-', alpha=0.3)
    
    # Calculate average success rates
    if enhanced_success:
        # Interpolate to common x-axis
        x = np.linspace(1, max_episodes, 1000)
        enhanced_interp = []
        for episodes, rates in enhanced_success:
            enhanced_interp.append(np.interp(x, episodes, rates))
        enhanced_avg = np.mean(enhanced_interp, axis=0)
        plt.plot(x, enhanced_avg, 'b-', linewidth=2, label='Enhanced DQN')
    
    if neuro_success:
        # Interpolate to common x-axis
        x = np.linspace(1, max_episodes, 1000)
        neuro_interp = []
        for episodes, rates in neuro_success:
            neuro_interp.append(np.interp(x, episodes, rates))
        neuro_avg = np.mean(neuro_interp, axis=0)
        plt.plot(x, neuro_avg, 'r-', linewidth=2, label='Neurosymbolic DQN')
    
    plt.title(f'Success Rate Comparison - {template.replace("_", " ").title()}')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate (100-episode moving average)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1.1)
    
    # Save success rate plot
    plt.savefig(os.path.join(output_dir, f'{template}_success_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Setup figure for scores
    plt.figure(figsize=(12, 8))
    
    # Plot scores for enhanced DQN
    for episodes, scores in enhanced_scores:
        plt.plot(episodes, scores, 'b-', alpha=0.3)
    
    # Plot scores for neurosymbolic DQN
    for episodes, scores in neuro_scores:
        plt.plot(episodes, scores, 'r-', alpha=0.3)
    
    # Calculate average scores
    if enhanced_scores:
        # Interpolate to common x-axis
        x = np.linspace(1, max_episodes, 1000)
        enhanced_interp = []
        for episodes, scores in enhanced_scores:
            enhanced_interp.append(np.interp(x, episodes, scores))
        enhanced_avg = np.mean(enhanced_interp, axis=0)
        plt.plot(x, enhanced_avg, 'b-', linewidth=2, label='Enhanced DQN')
    
    if neuro_scores:
        # Interpolate to common x-axis
        x = np.linspace(1, max_episodes, 1000)
        neuro_interp = []
        for episodes, scores in neuro_scores:
            neuro_interp.append(np.interp(x, episodes, scores))
        neuro_avg = np.mean(neuro_interp, axis=0)
        plt.plot(x, neuro_avg, 'r-', linewidth=2, label='Neurosymbolic DQN')
    
    plt.title(f'Score Comparison - {template.replace("_", " ").title()}')
    plt.xlabel('Episodes')
    plt.ylabel('Score (smoothed)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save score plot
    plt.savefig(os.path.join(output_dir, f'{template}_score.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_combined_plot(templates, output_dir):
    """Generate combined plot showing final success rates across all templates"""
    # Collection of final success rates
    enhanced_final = {}
    neuro_final = {}
    
    for template in templates:
        # Load success rates
        enhanced_path = os.path.join("experiments/neuro_symbolic_comparison", template, "enhanced_dqn")
        neuro_path = os.path.join("experiments/neuro_symbolic_comparison", template, "neurosymbolic_dqn")
        
        enhanced_logs = find_training_logs("experiments/neuro_symbolic_comparison", template, "enhanced_dqn")
        neuro_logs = find_training_logs("experiments/neuro_symbolic_comparison", template, "neurosymbolic_dqn")
        
        _, enhanced_success, _ = extract_metrics(enhanced_logs)
        _, neuro_success, _ = extract_metrics(neuro_logs)
        
        # Calculate final success rates (last 100 episodes)
        if enhanced_success:
            enhanced_finals = []
            for _, rates in enhanced_success:
                enhanced_finals.append(np.mean(rates[-100:]))
            enhanced_final[template] = np.mean(enhanced_finals)
        else:
            enhanced_final[template] = 0
        
        if neuro_success:
            neuro_finals = []
            for _, rates in neuro_success:
                neuro_finals.append(np.mean(rates[-100:]))
            neuro_final[template] = np.mean(neuro_finals)
        else:
            neuro_final[template] = 0
    
    # Create bar chart
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(templates))
    width = 0.35
    
    enhanced_values = [enhanced_final.get(t, 0) for t in templates]
    neuro_values = [neuro_final.get(t, 0) for t in templates]
    
    plt.bar(x - width/2, enhanced_values, width, label='Enhanced DQN', color='blue')
    plt.bar(x + width/2, neuro_values, width, label='Neurosymbolic DQN', color='red')
    
    plt.xlabel('Template')
    plt.ylabel('Final Success Rate')
    plt.title('Final Success Rate Comparison (Last 100 Episodes)')
    plt.xticks(x, [t.replace('_', ' ').title() for t in templates])
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(enhanced_values):
        plt.text(i - width/2, v + 0.05, f'{v:.2f}', ha='center')
    
    for i, v in enumerate(neuro_values):
        plt.text(i + width/2, v + 0.05, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Templates to process
    templates = ["basic_med", "sparse_med", "zipper_med", "bottleneck_med", "bottleneck_hard", "corridors_med"]
    
    # Create output directory
    output_dir = "plots/neuro_symbolic_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating neurosymbolic comparison plots...")
    
    # Process each template
    for template in templates:
        print(f"Processing template: {template}")
        
        # Find log files
        enhanced_logs = find_training_logs("experiments/neuro_symbolic_comparison", template, "enhanced_dqn")
        neuro_logs = find_training_logs("experiments/neuro_symbolic_comparison", template, "neurosymbolic_dqn")
        
        # Extract metrics
        enhanced_metrics = extract_metrics(enhanced_logs)
        neuro_metrics = extract_metrics(neuro_logs)
        
        # Generate plots
        template_output_dir = os.path.join(output_dir, template)
        plot_metrics(template, enhanced_metrics, neuro_metrics, template_output_dir)
    
    # Generate combined plot
    generate_combined_plot(templates, output_dir)
    
    print(f"All plots generated in: {output_dir}")
    print("Main plots:")
    for template in templates:
        print(f"  {template} success rate: {output_dir}/{template}/{template}_success_rate.png")
        print(f"  {template} score: {output_dir}/{template}/{template}_score.png")
    print(f"Overall comparison: {output_dir}/overall_comparison.png")

if __name__ == "__main__":
    main()
