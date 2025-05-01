import os
import subprocess
import argparse
import time
from graphing_utils import load_and_process_results, create_comparison_plots

def run_single_experiment(template, num_episodes, augmented, shaped, ksm_mode, seed, output_prefix):
    """Run a single experiment with specified parameters."""
    aug_str = "augmented" if augmented else "basic"
    shaping_str = "shaped" if shaped else "raw"
    ksm_str = ksm_mode if ksm_mode != "off" else "no_ksm"
    
    # Construct output directory name
    output_dir = f"{output_prefix}_{template}_{aug_str}_{shaping_str}_{ksm_str}_seed{seed}"
    
    # Check if this experiment has already been completed
    log_file = os.path.join(output_dir, 'training_log.csv')
    if os.path.exists(log_file):
        print(f"Experiment {output_dir} already completed, skipping...")
        return
    
    # Construct command
    cmd = [
        "python", "train_enhanced_dqn.py",
        "--template", template,
        "--episodes", str(num_episodes),
        "--output", output_prefix,
        "--seed", str(seed)
    ]
    
    # Add flags based on parameters
    if not augmented:
        cmd.append("--basic-state")
    if not shaped:
        cmd.append("--no-reward-shaping")
    if ksm_mode != "off":
        cmd.append("--ksm-mode")
        cmd.append(ksm_mode)
    
    # Run the command
    print(f"Running experiment: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_experiment_set(template, num_episodes, model_configs, num_seeds=5, output_prefix="experiment_results"):
    """Run a set of experiments with multiple seeds."""
    # Create base output directory
    os.makedirs("experiment_outputs", exist_ok=True)
    
    # Run each model configuration with each seed
    for config in model_configs:
        model_name = config['name']
        config['template'] = template  # Add template to config for later use
        
        print(f"\n=== Running {model_name} on {template} for {num_episodes} episodes ===\n")
        
        for seed in range(num_seeds):
            run_single_experiment(
                template=template,
                num_episodes=num_episodes,
                augmented=config['augmented'],
                shaped=config['shaped'],
                ksm_mode=config['ksm_mode'],
                seed=seed,
                output_prefix=output_prefix
            )
    
    # After all experiments, create comparative plots
    print("\n=== Generating comparative plots ===\n")
    results = load_and_process_results(output_prefix, model_configs, num_seeds=num_seeds)
    create_comparison_plots(
        results=results,
        output_dir="experiment_outputs",
        template_name=template,
        num_episodes=num_episodes,
        experiment_name=output_prefix
    )
    
    print(f"\n=== Experiment completed. Results saved to experiment_outputs/ ===\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run DQN experiments with multiple configurations')
    parser.add_argument('--experiment', type=str, required=True, 
                        choices=["dqn_variants", "ksm_variants", "ksm_with_baselines"],
                        help='Experiment type to run')
    parser.add_argument('--template', type=str, required=True,
                        choices=["basic_med", "sparse_med", "zipper_med", "bottleneck_med", "bottleneck_hard", "corridors_med"],
                        help='Environment template to use')
    parser.add_argument('--episodes', type=int, default=2000, 
                        help='Number of episodes per run')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of random seeds to run')
    
    args = parser.parse_args()
    
    # Define model configurations for each experiment type
    if args.experiment == "dqn_variants":
        # Baseline DQN vs. DQN with only reward shaping vs. DQN with only state augmentation vs. full enhanced DQN
        model_configs = [
            {'name': 'Baseline DQN', 'augmented': False, 'shaped': False, 'ksm_mode': 'off'},
            {'name': 'DQN + Reward Shaping', 'augmented': False, 'shaped': True, 'ksm_mode': 'off'},
            {'name': 'DQN + State Augmentation', 'augmented': True, 'shaped': False, 'ksm_mode': 'off'},
            {'name': 'Enhanced DQN', 'augmented': True, 'shaped': True, 'ksm_mode': 'off'}
        ]
        output_prefix = "dqn_variants"
        
    elif args.experiment == "ksm_variants":
        # Full enhanced DQN vs. full enhanced DQN + standard KSM vs. full enhanced DQN + adaptive KSM
        model_configs = [
            {'name': 'Enhanced DQN', 'augmented': True, 'shaped': True, 'ksm_mode': 'off'},
            {'name': 'Enhanced DQN + Standard KSM', 'augmented': True, 'shaped': True, 'ksm_mode': 'standard'},
            {'name': 'Enhanced DQN + Adaptive KSM', 'augmented': True, 'shaped': True, 'ksm_mode': 'adaptive'}
        ]
        output_prefix = "ksm_variants"
        
    elif args.experiment == "ksm_with_baselines":
        # Baseline DQN + adaptive KSM vs. DQN with only reward shaping + adaptive KSM vs. DQN with only state augmentation + adaptive KSM
        model_configs = [
            {'name': 'Baseline DQN + Adaptive KSM', 'augmented': False, 'shaped': False, 'ksm_mode': 'adaptive'},
            {'name': 'DQN + Reward Shaping + Adaptive KSM', 'augmented': False, 'shaped': True, 'ksm_mode': 'adaptive'},
            {'name': 'DQN + State Augmentation + Adaptive KSM', 'augmented': True, 'shaped': False, 'ksm_mode': 'adaptive'}
        ]
        output_prefix = "ksm_with_baselines"
    
    print(f"Starting experiment: {args.experiment} on template: {args.template} for {args.episodes} episodes with {args.seeds} seeds each")
    run_experiment_set(
        template=args.template,
        num_episodes=args.episodes,
        model_configs=model_configs,
        num_seeds=args.seeds,
        output_prefix=output_prefix
    )
