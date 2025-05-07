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
from dqn_agent_enhanced import DQNAgentEnhanced

# train DQN agent on custom LIFO corridors environment
def train_enhanced_dqn(template_name="basic_med", n_episodes=15000, max_t=200, eps_start=1.0, eps_end=0.005, eps_decay=0.998, 
                       render=False, checkpoint_dir='enhanced_results', eval_freq=100, eval_episodes=10,
                       use_augmented_state=True, use_reward_shaping=True, ksm_mode="off"):
    """
    template_name: name of template to use
    n_episodes (int): max number of training episodes
    max_t (int): max number of timesteps per episode
    eps_start (float): starting value of epsilon for epsilon-greedy action selection
    eps_end (float): minimum value of epsilon
    eps_decay (float): factor for decreasing epsilon
    render (bool): whether to render the environment during training
    checkpoint_dir (str): directory to save checkpoints and results
    eval_freq (int): frequency of evaluation during training
    eval_episodes (int): number of episodes to run during evaluation
    use_augmented_state (bool): whether to use augmented state representation
    use_reward_shaping (bool): whether to use reward shaping
    ksm_mode (str): Key Selection Metric mode ("off", "standard", or "adaptive")
    """
    # create directory for results
    aug_str = "augmented" if use_augmented_state else "basic"
    shaping_str = "shaped" if use_reward_shaping else "raw"
    ksm_str = ksm_mode if ksm_mode != "off" else "no_ksm"
    
    output_dir = f"{checkpoint_dir}_{template_name}_{aug_str}_{shaping_str}_{ksm_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    # log file for detailed stats
    log_file = os.path.join(output_dir, 'training_log.csv')
    
    # create environment
    env = TemplateLIFOCorridorsEnv(template_name=template_name, render_enabled=False, 
                                   verbose=False, use_reward_shaping=use_reward_shaping)
    
    # get state and action sizes
    state, _ = env.reset()
    # create a temporary agent to get the preprocessed state size
    temp_agent = DQNAgentEnhanced(0, 0, use_augmented_state=use_augmented_state, ksm_mode=ksm_mode)
    state_size = len(temp_agent.preprocess_state(state))
    action_size = env.action_space.n
    
    print(f"Template: {template_name}, State size: {state_size}, Action size: {action_size}")
    print(f"State representation: {aug_str}")
    print(f"Reward shaping: {shaping_str}")
    print(f"Key Selection Metric (KSM): {ksm_mode}")
    
    # create agent
    agent = DQNAgentEnhanced(state_size=state_size, action_size=action_size, 
                          seed=0, use_augmented_state=use_augmented_state,
                          ksm_mode=ksm_mode)
    
    # set template context for logging in adaptive KSM
    if ksm_mode == "adaptive":
        agent.set_template_context(template_name)
    
    # initialize epsilon
    eps = eps_start
    
    # lists and metrics to track progress
    scores = []
    scores_window = deque(maxlen=100)
    eps_history = []
    success_history = []
    win_episodes = []
    episode_steps = []
    success_rate_history = []
    wrong_key_attempts = []
    
    # for evaluation metrics
    eval_success_rates = []
    eval_scores = []
    eval_episodes_x = []
    
    # track time
    start_time = time.time()
    
    # create DF for logging
    log_data = {
        'episode': [], 
        'steps': [], 
        'score': [], 
        'epsilon': [], 
        'success': [], 
        'keys_collected': [], 
        'doors_opened': [],
        'wrong_key_attempts': [],
        'termination_reason': []
    }
    
    # progress bar
    pbar = tqdm(total=n_episodes, desc=f"Template: {template_name}", unit="ep")
    
    # TRAINING LOOP
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        score = 0
        success = False
        
        # episode stats
        steps = 0
        keys_collected = 0
        doors_opened = 0
        wrong_key_count = 0
        termination_reason = 'incomplete'
        
        # initialize total reward tracking for this episode
        agent.total_reward = 0
        
        for t in range(max_t):
            # select and perform action
            action = agent.act(state, eps)
            next_state, reward, done, _, info = env.step(action)
            
            # update stats
            steps = t + 1
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
            
            # store transition and learn
            agent.step(state, action, reward, next_state, done, info)
            
            # update state and score
            state = next_state
            score += reward
            agent.total_reward = score  # Update total reward for the episode
            
            # render if enabled
            if render:
                env.render()
                time.sleep(0.01) # slow down rendering
            
            if done:
                break
        
        # update metrics
        scores_window.append(score)
        scores.append(score)
        eps_history.append(eps)
        episode_steps.append(steps)
        success_history.append(1 if success else 0)
        wrong_key_attempts.append(wrong_key_count)
        
        # calc success rate and update agent's knowledge of it
        if len(success_history) >= 100:
            success_rate = sum(success_history[-100:]) / 100
        else:
            success_rate = sum(success_history) / len(success_history)
        
        agent.current_success_rate = success_rate
        success_rate_history.append(success_rate)
        
        # log data for this episode
        log_data['episode'].append(i_episode)
        log_data['steps'].append(steps)
        log_data['score'].append(score)
        log_data['epsilon'].append(eps)
        log_data['success'].append(success)
        log_data['keys_collected'].append(keys_collected)
        log_data['doors_opened'].append(doors_opened)
        log_data['wrong_key_attempts'].append(wrong_key_count)
        log_data['termination_reason'].append(termination_reason)
        
        # update epsilon
        eps = max(eps_end, eps_decay * eps)
        
        # check if episode was a win
        if success: win_episodes.append(i_episode)
        
        # update progress bar
        pbar.update(1)
        pbar.set_postfix({
            'Avg Score': f"{np.mean(scores_window):.2f}", 
            'Success Rate': f"{success_rate:.2f}",
            'Epsilon': f"{eps:.2f}"})
        
        # print progress
        if i_episode % 100 == 0:
            wrong_key_rate = sum(wrong_key_attempts[-100:]) / max(1, sum(episode_steps[-100:]))
            elapsed = time.time() - start_time
            print(f"\nEpisode {i_episode}/{n_episodes} | "
                  f"Avg Score: {np.mean(scores_window):.2f} | "
                  f"Success Rate: {success_rate:.2f} | "
                  f"Wrong Keys: {wrong_key_rate:.4f} | "
                  f"Epsilon: {eps:.3f} | "
                  f"Time: {elapsed:.1f}s")
        
        # periodically evaluate agent
        if i_episode % eval_freq == 0:
            eval_success_rate, eval_avg_score, eval_wrong_key_rate = evaluate_agent(agent, env, n_episodes=eval_episodes)
            
            eval_success_rates.append(eval_success_rate)
            eval_scores.append(eval_avg_score)
            eval_episodes_x.append(i_episode)
            
            print(f"\nEVALUATION at episode {i_episode}:")
            print(f"  Success Rate: {eval_success_rate:.2f}")
            print(f"  Average Score: {eval_avg_score:.2f}")
            print(f"  Wrong Key Rate: {eval_wrong_key_rate:.4f}")
            
            # save checkpoint
            agent.save(f"{output_dir}/dqn_checkpoint_{i_episode}.pth")
            
            # generate and save plots
            generate_training_plots(
                scores, eps_history, success_rate_history, wrong_key_attempts,
                episode_steps, eval_episodes_x, eval_success_rates, eval_scores,
                output_dir, i_episode, template_name, win_episodes, success_history)
            
            # save log to CSV
            pd.DataFrame(log_data).to_csv(log_file, index=False)
    
    # close progress bar
    pbar.close()
    
    # final save
    agent.save(f"{output_dir}/dqn_final.pth")
    
    # close environment
    env.close()
    
    # final plots
    generate_training_plots(
        scores, eps_history, success_rate_history, wrong_key_attempts,
        episode_steps, eval_episodes_x, eval_success_rates, eval_scores,
        output_dir, n_episodes, template_name, win_episodes, success_history)
    
    # final log save
    pd.DataFrame(log_data).to_csv(log_file, index=False)
    
    # rrint final statistics
    print("\nTraining complete!")
    print(f"Total win episodes: {len(win_episodes)}")
    if len(win_episodes) > 0:
        print(f"First win on episode: {win_episodes[0]}")
    
    # calc final success rates
    final_success_rate = sum(success_history[-min(100, len(success_history)):]) / min(100, len(success_history))
    print(f"Final success rate (last 100 episodes): {final_success_rate:.2f}")
    
    # calculate avg steps for successful episodes
    successful_steps = [s for s, success in zip(episode_steps, success_history) if success]
    if successful_steps:
        avg_steps = sum(successful_steps) / len(successful_steps)
        print(f"Average steps for successful episodes: {avg_steps:.1f}")
    
    # calculate wrong key attempt rate
    if episode_steps:
        wrong_key_rate = sum(wrong_key_attempts) / sum(episode_steps)
        print(f"Wrong key attempt rate: {wrong_key_rate:.4f} (attempts per step)")
    
    return scores, win_episodes, success_rate_history, wrong_key_attempts

# evaluate agent w/o exploration
def evaluate_agent(agent, env, n_episodes=10):
    success_count = 0
    total_score = 0
    total_steps = 0
    total_wrong_key_attempts = 0
    
    for i in range(n_episodes):
        state, _ = env.reset()
        score = 0
        steps = 0
        wrong_key_count = 0
        done = False
        
        while not done:
            action = agent.act(state, eps=0.0)  # No exploration
            next_state, reward, done, _, info = env.step(action)
            state = next_state
            score += reward
            steps += 1
            
            if 'wrong_key_attempt' in info:
                wrong_key_count += 1
            
            if done and 'success' in info and info['success']:
                success_count += 1
        
        total_score += score
        total_steps += steps
        total_wrong_key_attempts += wrong_key_count
    
    success_rate = success_count / n_episodes
    avg_score = total_score / n_episodes
    wrong_key_rate = total_wrong_key_attempts / max(1, total_steps)
    
    return success_rate, avg_score, wrong_key_rate

# generate training plots
def generate_training_plots(scores, eps_history, success_rate_history, wrong_key_attempts,
                           episode_steps, eval_episodes, eval_success_rates, eval_scores,
                           output_dir, episode_num, template_name, win_episodes, success_history):
    plt.figure(figsize=(15, 15))
    
    # training Scores
    plt.subplot(3, 2, 1)
    plt.plot(np.arange(len(scores)), scores)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # Zero line
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title(f'DQN Training Scores - {template_name} Template')
    
    # success rate
    plt.subplot(3, 2, 2)
    plt.plot(np.arange(len(success_rate_history)), success_rate_history)
    plt.ylabel('Success Rate')
    plt.xlabel('Episode #')
    plt.title('Training Success Rate')
    
    # epsilon decay
    plt.subplot(3, 2, 3)
    plt.plot(np.arange(len(eps_history)), eps_history)
    plt.ylabel('Epsilon')
    plt.xlabel('Episode #')
    plt.title('Epsilon Decay')
    
    # wrong key attempts
    plt.subplot(3, 2, 4)
    if len(wrong_key_attempts) > 0:
        # wrong key rate over time (sliding window)
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
    
    # episode length
    plt.subplot(3, 2, 5)
    if len(episode_steps) > 0:
        # avg episode length over time (sliding window)
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
    
    # evaluation metrics
    plt.subplot(3, 2, 6)
    if eval_episodes and eval_success_rates and eval_scores:
        # twin axis for scores and success rates
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        #success rates on left axis
        line1 = ax1.plot(eval_episodes, eval_success_rates, 'b-', label='Success Rate')
        ax1.set_xlabel('Episode #')
        ax1.set_ylabel('Success Rate', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # scores on right axis
        line2 = ax2.plot(eval_episodes, eval_scores, 'r-', label='Avg Score')
        ax2.set_ylabel('Average Score', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        plt.title('Evaluation Metrics')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_progress_{episode_num}.png")
    plt.close()
    
    # additional specialized plots:
    
    # moving average of scores (smoother)
    plt.figure(figsize=(10, 6))
    window_size = min(100, len(scores))
    if len(scores) >= window_size:
        scores_smoothed = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(len(scores_smoothed)), scores_smoothed)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # Zero line
        plt.ylabel('Score (Moving Average)')
        plt.xlabel('Episode #')
        plt.title(f'Smoothed DQN Training Scores - {template_name} Template (Window Size: {window_size})')
        plt.savefig(f"{output_dir}/smoothed_scores_{episode_num}.png")
    plt.close()
    
    # distribution of episode lengths for successful vs. failed episodes
    plt.figure(figsize=(12, 6))
    if len(episode_steps) > 0 and len(success_history) > 0:
        # lists of episode lengths for successful and failed episodes
        success_lengths = []
        failure_lengths = []
        
        for i in range(len(episode_steps)):
            if i < len(success_history) and success_history[i] == 1:
                success_lengths.append(episode_steps[i])
            else:
                failure_lengths.append(episode_steps[i])
        
        if success_lengths and failure_lengths:
            plt.hist([success_lengths, failure_lengths], bins=20, label=['Success', 'Failure'], alpha=0.7)
            plt.xlabel('Episode Length (steps)')
            plt.ylabel('Count')
            plt.legend()
            plt.title(f'Distribution of Episode Lengths - {template_name} Template')
            plt.savefig(f"{output_dir}/length_distribution_{episode_num}.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN agent on Enhanced LIFO Corridors Environment')
    parser.add_argument('--template', type=str, default="basic_med", choices=["basic_med", "sparse_med", "zipper_med", "bottleneck_med", "bottleneck_hard", "corridors_med"], help='Template to use')
    parser.add_argument('--episodes', type=int, default=4000, help='Number of episodes')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--output', type=str, default='enhanced_results', help='Output directory prefix')
    parser.add_argument('--eval-freq', type=int, default=100, help='Evaluation frequency')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--basic-state', action='store_true', help='Use basic state representation (no augmentation)')
    parser.add_argument('--no-reward-shaping', action='store_true', help='Disable reward shaping')
    parser.add_argument('--ksm-mode', type=str, default="off", choices=["off", "standard", "adaptive"], help='Key Selection Metric mode')
    
    args = parser.parse_args()
    
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    print(f"Starting DQN training for Enhanced LIFO environment...")
    print(f"Template: {args.template}")
    print(f"Training for {args.episodes} episodes")
    print(f"State representation: {'Basic' if args.basic_state else 'Augmented'}")
    print(f"Reward shaping: {'OFF' if args.no_reward_shaping else 'ON'}")
    print(f"Key Selection Metric (KSM): {args.ksm_mode}")
    
    # run training with parameters
    scores, win_episodes, success_rates, wrong_key_attempts = train_enhanced_dqn(
        template_name=args.template,
        n_episodes=args.episodes, 
        render=args.render,
        checkpoint_dir=args.output,
        eval_freq=args.eval_freq,
        use_augmented_state=not args.basic_state,
        use_reward_shaping=not args.no_reward_shaping,
        ksm_mode=args.ksm_mode
    )
