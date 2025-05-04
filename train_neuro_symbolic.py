# train_neuro_symbolic.py

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import time
import argparse
import pandas as pd
import random
from tqdm import tqdm

from template_lifo_corridors import TemplateLIFOCorridorsEnv
from neurosymbolic_agent import NeurosymbolicDQNAgent

def train_neuro_symbolic(template_name="basic_med", n_episodes=4000, max_t=200, 
                         eps_start=1.0, eps_end=0.005, eps_decay=0.998, 
                         render=False, checkpoint_dir='neuro_symbolic_results',
                         eval_freq=100, eval_episodes=10, seed=0,
                         symbolic_guidance_weight=0.65,
                         use_base_dqn=False,
                         use_reward_shaping=True,
                         gradual_guidance_decrease=False,
                         min_guidance_weight=0.3,
                         guidance_decay=0.9999):
    """Train Neurosymbolic DQN agent on LIFO Corridors environment."""
    
    # Create output directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Log file for detailed statistics
    log_file = os.path.join(checkpoint_dir, 'training_log.csv')
    
    # Create environment
    env = TemplateLIFOCorridorsEnv(template_name=template_name, render_enabled=False, 
                                   verbose=False, use_reward_shaping=use_reward_shaping)
    
    # Get state and action sizes
    state, _ = env.reset()
    
    # Create a temporary agent to get the preprocessed state size
    temp_agent = NeurosymbolicDQNAgent(0, 0, use_augmented_state=not use_base_dqn)
    state_size = len(temp_agent.preprocess_state(state))
    action_size = env.action_space.n
    
    # Print configuration
    dqn_type = "Base DQN" if use_base_dqn else "Enhanced DQN"
    guidance_type = "Gradually decreasing" if gradual_guidance_decrease else "Fixed"
    reward_shaping_str = "enabled" if use_reward_shaping else "disabled"
    
    print(f"Template: {template_name}, State size: {state_size}, Action size: {action_size}")
    print(f"Neurosymbolic {dqn_type} with {guidance_type} guidance weight: {symbolic_guidance_weight}")
    print(f"Reward shaping: {reward_shaping_str}")
    if gradual_guidance_decrease:
        print(f"Guidance weight will decrease to {min_guidance_weight} with decay factor {guidance_decay}")
    
    # Create agent
    agent = NeurosymbolicDQNAgent(
        state_size=state_size, 
        action_size=action_size, 
        seed=seed, 
        use_augmented_state=not use_base_dqn,
        ksm_mode="off",  # KSM is turned off for neurosymbolic agent
        symbolic_guidance_weight=symbolic_guidance_weight,
        use_base_dqn=use_base_dqn,
        gradual_guidance_decrease=gradual_guidance_decrease,
        min_guidance_weight=min_guidance_weight,
        guidance_decay=guidance_decay
    )
    
    # Set template context for logging
    agent.set_template_context(template_name)
    
    # Initialize epsilon
    eps = eps_start
    
    # Lists and metrics to track progress
    scores = []
    scores_window = deque(maxlen=100)
    eps_history = []
    success_history = []
    win_episodes = []
    episode_steps = []
    success_rate_history = []
    wrong_key_attempts = []
    guidance_weight_history = []
    
    # For symbolic decision tracking
    neural_decisions = []
    guided_decisions = []
    
    # Track time
    start_time = time.time()
    
    # Create DataFrame for logging
    log_data = {
        'episode': [], 
        'steps': [], 
        'score': [], 
        'epsilon': [], 
        'success': [], 
        'keys_collected': [], 
        'doors_opened': [],
        'wrong_key_attempts': [],
        'termination_reason': [],
        'guidance_weight': [],
        'neural_decision_rate': [],
        'guided_decision_rate': []
    }
    
    # Define progress bar
    pbar = tqdm(total=n_episodes, desc=f"Template: {template_name}", unit="ep")
    
    # Training loop
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        score = 0
        success = False
        
        # Episode statistics
        steps = 0
        keys_collected = 0
        doors_opened = 0
        wrong_key_count = 0
        termination_reason = 'incomplete'
        
        # Reset tracking variables
        decision_counts = {'neural': 0, 'guided': 0}
        
        # Initialize total reward tracking for this episode
        agent.dqn_agent.total_reward = 0
        
        for t in range(max_t):
            # Select and perform action
            action = agent.act(state, eps)
            next_state, reward, done, _, info = env.step(action)
            
            # Update statistics
            steps = t + 1
            
            # Track symbolic decision type
            if agent.decision_history and agent.decision_history[-1]['mode'] == 'guided':
                decision_counts['guided'] += 1
            else:
                decision_counts['neural'] += 1
            
            if 'collected_key' in info:
                keys_collected += 1
                
            if 'opened_door' in info:
                doors_opened += 1
                
            if 'wrong_key_attempt' in info:
                wrong_key_count += 1
                
            if 'success' in info and info['success']:
                success = True
                
            if 'terminated_reason' in info:
                termination_reason = info['terminated_reason']
            
            # Store transition and learn
            agent.step(state, action, reward, next_state, done, info)
            
            # Update state and score
            state = next_state
            score += reward
            agent.dqn_agent.total_reward = score  # Update total reward for the episode
            
            # Render if enabled
            if render:
                env.render()
                time.sleep(0.01)  # Slow down rendering
            
            if done:
                break
        
        # Get current guidance weight (may have been decreased if using gradual decrease)
        guidance_weight = agent.symbolic_guidance_weight
        guidance_weight_history.append(guidance_weight)
        
        # Calculate decision rates
        total_decisions = sum(decision_counts.values())
        neural_rate = decision_counts['neural'] / total_decisions if total_decisions > 0 else 0
        guided_rate = decision_counts['guided'] / total_decisions if total_decisions > 0 else 0
        
        neural_decisions.append(neural_rate)
        guided_decisions.append(guided_rate)
        
        # Update metrics
        scores_window.append(score)
        scores.append(score)
        eps_history.append(eps)
        episode_steps.append(steps)
        success_history.append(1 if success else 0)
        wrong_key_attempts.append(wrong_key_count)
        
        # Calculate success rate and update agent's knowledge of it
        if len(success_history) >= 100:
            success_rate = sum(success_history[-100:]) / 100
        else:
            success_rate = sum(success_history) / len(success_history)
        
        agent.dqn_agent.current_success_rate = success_rate
        success_rate_history.append(success_rate)
        
        # Log data for this episode
        log_data['episode'].append(i_episode)
        log_data['steps'].append(steps)
        log_data['score'].append(score)
        log_data['epsilon'].append(eps)
        log_data['success'].append(success)
        log_data['keys_collected'].append(keys_collected)
        log_data['doors_opened'].append(doors_opened)
        log_data['wrong_key_attempts'].append(wrong_key_count)
        log_data['termination_reason'].append(termination_reason)
        log_data['guidance_weight'].append(guidance_weight)
        log_data['neural_decision_rate'].append(neural_rate)
        log_data['guided_decision_rate'].append(guided_rate)
        
        # Update epsilon
        eps = max(eps_end, eps_decay * eps)
        
        # Check if episode was a win
        if success:
            win_episodes.append(i_episode)
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            'Avg Score': f"{np.mean(scores_window):.2f}", 
            'Success Rate': f"{success_rate:.2f}",
            'G-Weight': f"{guidance_weight:.2f}"
        })
        
        # Print progress
        if i_episode % 100 == 0:
            wrong_key_rate = sum(wrong_key_attempts[-100:]) / max(1, sum(episode_steps[-100:]))
            elapsed = time.time() - start_time
            print(f"\nEpisode {i_episode}/{n_episodes} | "
                  f"Avg Score: {np.mean(scores_window):.2f} | "
                  f"Success Rate: {success_rate:.2f} | "
                  f"Wrong Keys: {wrong_key_rate:.4f} | "
                  f"Epsilon: {eps:.3f} | "
                  f"G-Weight: {guidance_weight:.2f} | "
                  f"Time: {elapsed:.1f}s")
            
            # Print symbolic decision stats
            neural_avg = np.mean(neural_decisions[-100:]) if neural_decisions else 0
            guided_avg = np.mean(guided_decisions[-100:]) if guided_decisions else 0
            print(f"Decision Rates: Neural {neural_avg:.2f}, Guided {guided_avg:.2f}")
        
        # Periodically save checkpoint and generate plots
        if i_episode % eval_freq == 0:
            # Save checkpoint
            agent.save(f"{checkpoint_dir}/dqn_checkpoint_{i_episode}.pth")
            
            # Generate and save plots
            generate_training_plots(
                scores, eps_history, success_rate_history, wrong_key_attempts,
                episode_steps, guidance_weight_history, neural_decisions, guided_decisions,
                checkpoint_dir, i_episode, template_name, win_episodes, success_history
            )
            
            # Save log to CSV
            pd.DataFrame(log_data).to_csv(log_file, index=False)
    
    # Close progress bar
    pbar.close()
    
    # Final save
    agent.save(f"{checkpoint_dir}/dqn_final.pth")
    
    # Close environment
    env.close()
    
    # Final plots
    generate_training_plots(
        scores, eps_history, success_rate_history, wrong_key_attempts,
        episode_steps, guidance_weight_history, neural_decisions, guided_decisions,
        checkpoint_dir, n_episodes, template_name, win_episodes, success_history
    )
    
    # Final log save
    pd.DataFrame(log_data).to_csv(log_file, index=False)
    
    # Print final statistics
    print("\nTraining complete!")
    print(f"Total win episodes: {len(win_episodes)}")
    if len(win_episodes) > 0:
        print(f"First win on episode: {win_episodes[0]}")
    
    # Calculate final success rates
    final_success_rate = sum(success_history[-min(100, len(success_history)):]) / min(100, len(success_history))
    print(f"Final success rate (last 100 episodes): {final_success_rate:.2f}")
    
    # Calculate average steps for successful episodes
    successful_steps = [s for s, success in zip(episode_steps, success_history) if success]
    if successful_steps:
        avg_steps = sum(successful_steps) / len(successful_steps)
        print(f"Average steps for successful episodes: {avg_steps:.1f}")
    
    # Calculate wrong key attempt rate
    if episode_steps:
        wrong_key_rate = sum(wrong_key_attempts) / sum(episode_steps)
        print(f"Wrong key attempt rate: {wrong_key_rate:.4f} (attempts per step)")
    
    return scores, win_episodes, success_rate_history, wrong_key_attempts

def generate_training_plots(scores, eps_history, success_rate_history, wrong_key_attempts,
                           episode_steps, guidance_weight_history, neural_decisions, guided_decisions,
                           output_dir, episode_num, template_name, win_episodes, success_history):
    """Generate plots for neurosymbolic agent training."""
    plt.figure(figsize=(15, 20))
    
    # Plot 1: Training Scores
    plt.subplot(4, 2, 1)
    plt.plot(np.arange(len(scores)), scores)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # Zero line
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title(f'Neurosymbolic DQN Training Scores - {template_name} Template')
    
    # Plot 2: Success Rate
    plt.subplot(4, 2, 2)
    plt.plot(np.arange(len(success_rate_history)), success_rate_history)
    plt.ylabel('Success Rate')
    plt.xlabel('Episode #')
    plt.title('Training Success Rate')
    
    # Plot 3: Epsilon Decay
    plt.subplot(4, 2, 3)
    plt.plot(np.arange(len(eps_history)), eps_history)
    plt.ylabel('Epsilon')
    plt.xlabel('Episode #')
    plt.title('Epsilon Decay')
    
    # Plot 4: Wrong Key Attempts
    plt.subplot(4, 2, 4)
    if len(wrong_key_attempts) > 0:
        # Calculate wrong key rate over time (sliding window)
        window_size = min(100, len(wrong_key_attempts))
        wrong_key_rates = []
        for i in range(len(wrong_key_attempts)):
            if i < window_size - 1:
                rate = sum(wrong_key_attempts[:i+1]) / max(1, sum(episode_steps[:i+1]))
            else:
                rate = sum(wrong_key_attempts[i-window_size+1:i+1]) / max(1, sum(episode_steps[i-window_size+1:i+1]))
            wrong_key_rates.append(rate)
        
        plt.plot(np.arange(len(wrong_key_rates)), wrong_key_rates)
        plt.ylabel('Wrong Key Rate')
        plt.xlabel('Episode #')
        plt.title(f'Wrong Key Attempt Rate (per step, window={window_size})')
    
    # Plot 5: Episode Length
    plt.subplot(4, 2, 5)
    if len(episode_steps) > 0:
        # Calculate average episode length over time (sliding window)
        window_size = min(100, len(episode_steps))
        avg_steps = []
        for i in range(len(episode_steps)):
            if i < window_size - 1:
                avg = sum(episode_steps[:i+1]) / (i+1)
            else:
                avg = sum(episode_steps[i-window_size+1:i+1]) / window_size
            avg_steps.append(avg)
        
        plt.plot(np.arange(len(avg_steps)), avg_steps)
        plt.ylabel('Average Steps')
        plt.xlabel('Episode #')
        plt.title(f'Average Episode Length (window={window_size})')
    
    # Plot 6: Guidance Weight History
    plt.subplot(4, 2, 6)
    if len(guidance_weight_history) > 0:
        plt.plot(np.arange(len(guidance_weight_history)), guidance_weight_history)
        plt.ylabel('Guidance Weight')
        plt.xlabel('Episode #')
        plt.title('Symbolic Guidance Weight')
    
    # Plot 7: Neural vs Guided Decision Rates
    plt.subplot(4, 2, 7)
    if len(neural_decisions) > 0 and len(guided_decisions) > 0:
        plt.plot(np.arange(len(neural_decisions)), neural_decisions, label='Neural')
        plt.plot(np.arange(len(guided_decisions)), guided_decisions, label='Guided')
        plt.ylabel('Decision Rate')
        plt.xlabel('Episode #')
        plt.title('Neural vs Guided Decision Rates')
        plt.legend()
    
    # Plot 8: Specialized Stats
    plt.subplot(4, 2, 8)
    # Smoothed success rate
    window_size = min(50, len(success_rate_history))
    if len(success_rate_history) >= window_size:
        smoothed_rate = np.convolve(success_rate_history, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(len(smoothed_rate)), smoothed_rate)
        plt.ylabel('Smoothed Success Rate')
        plt.xlabel('Episode #')
        plt.title(f'Smoothed Success Rate (window={window_size})')
    else:
        plt.text(0.5, 0.5, "Neurosymbolic DQN\nDecision Analysis", 
                horizontalalignment='center', verticalalignment='center', 
                transform=plt.gca().transAxes, fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_progress_{episode_num}.png")
    plt.close()
    
    # Moving average of scores (smoother)
    plt.figure(figsize=(10, 6))
    window_size = min(100, len(scores))
    if len(scores) >= window_size:
        scores_smoothed = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(len(scores_smoothed)), scores_smoothed)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # Zero line
        plt.ylabel('Score (Moving Average)')
        plt.xlabel('Episode #')
        plt.title(f'Smoothed Neurosymbolic DQN Scores - {template_name} (Window Size: {window_size})')
        plt.savefig(f"{output_dir}/smoothed_scores_{episode_num}.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Neurosymbolic DQN agent on LIFO Corridors Environment')
    parser.add_argument('--template', type=str, default="basic_med", 
                        choices=["basic_med", "sparse_med", "zipper_med", "bottleneck_med", "bottleneck_hard", "corridors_med"],
                        help='Template to use')
    parser.add_argument('--episodes', type=int, default=4000, help='Number of episodes')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--output', type=str, default='neuro_symbolic_results', help='Output directory')
    parser.add_argument('--eval-freq', type=int, default=100, help='Evaluation frequency')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--guidance-weight', type=float, default=0.65, help='Weight for symbolic guidance (0-1)')
    parser.add_argument('--use-base-dqn', action='store_true', help='Use base DQN (no state augmentation or KSM)')
    parser.add_argument('--no-reward-shaping', action='store_true', help='Disable reward shaping')
    parser.add_argument('--gradual-guidance', action='store_true', help='Gradually decrease guidance weight')
    parser.add_argument('--min-guidance', type=float, default=0.3, help='Minimum guidance weight')
    parser.add_argument('--guidance-decay', type=float, default=0.9999, help='Guidance weight decay factor')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    print(f"Starting Neurosymbolic DQN training for LIFO environment...")
    print(f"Template: {args.template}")
    print(f"Training for {args.episodes} episodes")
    print(f"Symbolic guidance weight: {args.guidance_weight}")
    print(f"DQN type: {'Base' if args.use_base_dqn else 'Enhanced'}")
    print(f"Reward shaping: {'disabled' if args.no_reward_shaping else 'enabled'}")
    print(f"Guidance weight: {'Gradually decreasing' if args.gradual_guidance else 'Fixed'}")
    
    # Run training with parameters
    scores, win_episodes, success_rates, wrong_key_attempts = train_neuro_symbolic(
        template_name=args.template,
        n_episodes=args.episodes, 
        render=args.render,
        checkpoint_dir=args.output,
        eval_freq=args.eval_freq,
        seed=args.seed,
        symbolic_guidance_weight=args.guidance_weight,
        use_base_dqn=args.use_base_dqn,
        use_reward_shaping=not args.no_reward_shaping,
        gradual_guidance_decrease=args.gradual_guidance,
        min_guidance_weight=args.min_guidance,
        guidance_decay=args.guidance_decay
    )
