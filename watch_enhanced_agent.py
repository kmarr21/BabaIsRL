import numpy as np
import torch
import time
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from tqdm import tqdm

from template_lifo_corridors import TemplateLIFOCorridorsEnv
from dqn_agent_enhanced import DQNAgentEnhanced

def watch_enhanced_agent(model_path, template_name="basic_med", num_episodes=10, delay=0.2, save_dir=None, verbose=True):
    """Watch a trained agent navigate the enhanced LIFO environment with detailed metrics."""
    
    # Create output directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Create environment
    env = TemplateLIFOCorridorsEnv(template_name=template_name, render_enabled=True, verbose=False)
    
    # Get state and action sizes
    state, _ = env.reset()
    temp_agent = DQNAgentEnhanced(0, 0)
    state_size = len(temp_agent.preprocess_state(state))
    action_size = env.action_space.n
    
    # Create agent
    agent = DQNAgentEnhanced(state_size=state_size, action_size=action_size)
    
    # Load trained model
    agent.load(model_path)
    print(f"Loaded model from {model_path}")
    
    # Metrics to track
    episode_data = {
        'episode': [],
        'steps': [],
        'score': [],
        'success': [],
        'keys_collected': [],
        'doors_opened': [],
        'wrong_key_attempts': [],
        'termination_reason': []
    }
    
    # For tracking action distributions
    action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'STAY']
    all_action_counts = {name: 0 for name in action_names}
    
    # Define color mapping for keys and doors
    key_door_colors = {
        0: "Orange", 
        1: "Purple"
    }
    
    # Watch the agent
    for i_episode in range(1, num_episodes+1):
        state, _ = env.reset()
        score = 0
        step_count = 0
        
        # Episode tracking
        success = False
        keys_collected = 0
        doors_opened = 0
        wrong_key_count = 0
        termination_reason = 'incomplete'
        
        # Action count for stats
        action_counts = [0, 0, 0, 0, 0]  # up, right, down, left, stay
        
        if verbose:
            template_info = env.templates[template_name]
            template_display_name = template_info["name"]
            print(f"\nEpisode {i_episode}/{num_episodes} - Template: {template_display_name}")
            time.sleep(0.5)  # Pause before starting
        
        while True:
            # Select action (no exploration)
            action = agent.act(state, eps=0.0)
            action_counts[action] += 1
            
            # Take action in environment
            next_state, reward, done, _, info = env.step(action)
            
            # Track metrics
            step_count += 1
            if 'collected_key' in info:
                keys_collected += 1
                if verbose:
                    collected_key = info['collected_key']
                    color_name = key_door_colors[collected_key]
                    print(f"Collected {color_name} key! +1.0 reward")
                
            if 'opened_door' in info:
                doors_opened += 1
                if verbose:
                    opened_door = info['opened_door']
                    color_name = key_door_colors[opened_door]
                    print(f"Opened {color_name} door! +2.0 reward")
                
            if 'wrong_key_attempt' in info:
                wrong_key_count += 1
                if verbose:
                    print(f"Wrong key! -1.0 reward")
                
            if 'success' in info and info['success']:
                success = True
                
            if 'terminated_reason' in info:
                termination_reason = info['terminated_reason']
            
            # Update state and score
            state = next_state
            score += reward
            
            # Render environment
            env.render()
            
            # Add delay for better visualization
            time.sleep(delay)
            
            # Print step information
            if verbose:
                key_stack_info = f"Key stack: {env.key_stack}" if hasattr(env, 'key_stack') else ""
                print(f"Step {step_count}: Action={action_names[action]}, Reward={reward:.2f}, "
                    f"Keys={np.sum(state['key_status'])}/2, Doors={np.sum(state['door_status'])}/2 {key_stack_info}", 
                    end="\r")
            
            if done:
                # Print appropriate message based on outcome
                if verbose:
                    if success:
                        print(f"\nEpisode {i_episode} completed successfully in {step_count} steps! Score: {score:.2f}")
                    elif termination_reason == 'timeout':
                        print(f"\nEpisode {i_episode} timed out after {step_count} steps. Score: {score:.2f}")
                    else:
                        print(f"\nEpisode {i_episode} failed after {step_count} steps. Score: {score:.2f}")
                    
                    # Print action distribution
                    print("Action distribution:")
                    for i, (name, count) in enumerate(zip(action_names, action_counts)):
                        print(f"  {name}: {count} ({count/step_count*100:.1f}%)")
                        all_action_counts[name] += count
                
                break
        
        # Record episode data
        episode_data['episode'].append(i_episode)
        episode_data['steps'].append(step_count)
        episode_data['score'].append(score)
        episode_data['success'].append(success)
        episode_data['keys_collected'].append(keys_collected)
        episode_data['doors_opened'].append(doors_opened)
        episode_data['wrong_key_attempts'].append(wrong_key_count)
        episode_data['termination_reason'].append(termination_reason)
    
    # Close environment
    env.close()
    
    # Calculate summary statistics
    success_rate = sum(episode_data['success']) / num_episodes
    avg_steps = sum(episode_data['steps']) / num_episodes
    avg_score = sum(episode_data['score']) / num_episodes
    wrong_key_rate = sum(episode_data['wrong_key_attempts']) / max(1, sum(episode_data['steps']))
    
    # Print summary
    template_info = env.templates[template_name]
    template_display_name = template_info["name"]
    print("\n===== Summary Statistics =====")
    print(f"Template: {template_display_name}")
    print(f"Success Rate: {success_rate:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Wrong Key Rate: {wrong_key_rate:.4f} attempts per step")
    
    # Print overall action distribution
    total_actions = sum(all_action_counts.values())
    print("\nOverall Action Distribution:")
    for name, count in all_action_counts.items():
        if total_actions > 0:
            print(f"  {name}: {count} ({count/total_actions*100:.1f}%)")
    
    # Save results if directory provided
    if save_dir:
        # Save episode data to CSV
        df = pd.DataFrame(episode_data)
        df.to_csv(os.path.join(save_dir, 'test_results.csv'), index=False)
        
        # Create visualization of results
        plt.figure(figsize=(15, 10))
        
        # Plot steps per episode
        plt.subplot(2, 2, 1)
        plt.bar(range(1, num_episodes+1), episode_data['steps'], color=['g' if s else 'r' for s in episode_data['success']])
        plt.axhline(y=avg_steps, color='black', linestyle='--', label=f'Avg: {avg_steps:.1f}')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title(f'Steps per Episode - {template_display_name}')
        plt.legend()
        
        # Plot scores per episode
        plt.subplot(2, 2, 2)
        plt.bar(range(1, num_episodes+1), episode_data['score'], color=['g' if s else 'r' for s in episode_data['success']])
        plt.axhline(y=avg_score, color='black', linestyle='--', label=f'Avg: {avg_score:.2f}')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Score per Episode')
        plt.legend()
        
        # Pie chart of outcomes
        plt.subplot(2, 2, 3)
        termination_counts = {}
        for reason in episode_data['termination_reason']:
            termination_counts[reason] = termination_counts.get(reason, 0) + 1
        
        labels = list(termination_counts.keys())
        sizes = list(termination_counts.values())
        colors = ['green' if label == 'success' else 'red' for label in labels]
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        plt.title('Episode Outcomes')
        
        # Bar chart of action distribution
        plt.subplot(2, 2, 4)
        actions = list(all_action_counts.keys())
        counts = list(all_action_counts.values())
        plt.bar(actions, counts)
        plt.ylabel('Count')
        plt.title('Action Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'test_results.png'))
        
        # Create additional plot for wrong key attempts
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, num_episodes+1), episode_data['wrong_key_attempts'], color='orange')
        plt.axhline(y=np.mean(episode_data['wrong_key_attempts']), color='black', linestyle='--', 
                   label=f'Avg: {np.mean(episode_data["wrong_key_attempts"]):.2f}')
        plt.xlabel('Episode')
        plt.ylabel('Wrong Key Attempts')
        plt.title('Wrong Key Attempts per Episode')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'wrong_key_attempts.png'))
        
        print(f"Results saved to {save_dir}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Watch a trained DQN agent in enhanced LIFO environment')
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to model file')
    parser.add_argument('--template', type=str, default="basic_med", 
                       choices=["basic_med", "sparse_med", "zipper_med", "bottleneck_med", "corridors_med"],
                       help='Template to use')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to watch')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay between steps (seconds)')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output directory for results (default: test_results_{template})')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbose output')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Set default output directory if not specified
    if args.output is None:
        args.output = f"test_results_{args.template}"
    
    template_info = TemplateLIFOCorridorsEnv(template_name=args.template).templates[args.template]
    template_display_name = template_info["name"]
    
    print(f"Watching agent from model: {args.model}")
    print(f"Template: {args.template} ({template_display_name})")
    print(f"Running for {args.episodes} episodes with {args.delay}s delay")
    
    watch_enhanced_agent(
        args.model, 
        template_name=args.template,
        num_episodes=args.episodes, 
        delay=args.delay, 
        save_dir=args.output, 
        verbose=not args.quiet
    )
