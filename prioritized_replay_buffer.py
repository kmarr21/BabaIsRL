import numpy as np
import torch
import random
from collections import namedtuple

# define transition tuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# prioritized experience replay (PER) buffer for storing and sampling transitions
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=100000, device="cpu"):
        """
        capacity (int): max size of buffer
        alpha (float): how much prioritization to use (0 = no prioritization, 1 = full prioritization)
        beta_start (float): start value of importance-sampling correction (0 = no correction, 1 = full correction)
        beta_end (float): final value of beta after annealing
        beta_frames (int): num of frames over which to anneal beta
        device (str): device to store tensors on
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.device = device
        
        # current position in buffer
        self.position = 0
        # current size of buffer (for before it's full!)
        self.size = 0
        # frame counter for beta annealing
        self.frame = 1
        
        # buffers for experiences
        self.memory = []
        # priorities w/ small epsilon to avoid 0 probabilities
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
        # track successful episodes separately
        self.success_memory = []
        self.success_count = 0
        self.max_success_episodes = 250 

    # store a transition in the buffer
    def push(self, state, action, next_state, reward, done, is_success=False):
        # find max priority (use 1 for new experiences if buffer empty)
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        transition = Transition(state, action, next_state, reward, done)
        
        # for successful episodes, store separately
        if is_success and done:
            if self.success_count < self.max_success_episodes:
                self.success_memory.append(transition)
                self.success_count += 1
            elif self.success_count >= self.max_success_episodes:
                # if success buffer full, replace a random successful episode
                replace_idx = random.randint(0, self.max_success_episodes - 1)
                self.success_memory[replace_idx] = transition
        
        # store in main memory
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
            self.size += 1
        else:
            self.memory[self.position] = transition
        
        # update priority
        self.priorities[self.position] = max_priority
        
        # update position
        self.position = (self.position + 1) % self.capacity

    # sample a batch of transitions (success_bias = probability of sampling from successful episodes)
    def sample(self, batch_size, success_bias=0.4):
        # increase frame counter (for beta annealing)
        self.frame += 1
        
        # decide whether to sample from success memory
        use_success = (random.random() < success_bias) and self.success_count > 0
        
        if use_success:
            # sample from successful episodes
            indices = np.random.choice(self.success_count, min(batch_size, self.success_count), replace=False)
            batch = [self.success_memory[idx] for idx in indices]
            # use uniform weights for success samples
            weights = torch.ones(len(batch), device=self.device)
            
            # if need more samples, get them from regular memory
            if len(batch) < batch_size and self.size > 0:
                regular_batch_size = batch_size - len(batch)
                regular_indices, regular_weights, _ = self._sample_proportional(regular_batch_size)
                
                batch.extend([self.memory[idx] for idx in regular_indices])
                # combine weights
                weights = torch.cat([weights, regular_weights])
                indices = np.concatenate([indices, regular_indices])
            
        else:
            # sample from regular memory based on priorities
            indices, weights, probs = self._sample_proportional(batch_size)
            batch = [self.memory[idx] for idx in indices]
        
        # convert batch to tensor format
        states, actions, next_states, rewards, dones = zip(*batch)
        
        # stack states and next_states if they're tensors
        if isinstance(states[0], torch.Tensor):
            states = torch.stack(states)
            next_states = torch.stack(next_states)
        
        # convert other elements to tensors
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float, device=self.device)
        
        return (states, actions, next_states, rewards, dones), weights, indices

    # sample indices based on priorities
    def _sample_proportional(self, batch_size):
        if self.size < batch_size:
            # if buffer isn't full yet, sample from what we have
            indices = np.random.choice(self.size, self.size, replace=False)
            # pad with random indices if needed
            if self.size < batch_size:
                indices = np.concatenate([indices, np.random.choice(self.size, batch_size - self.size, replace=True)])
        else:
            # sample from priorities
            priorities = self.priorities[:self.size]
            probs = priorities ** self.alpha
            probs /= probs.sum()
            
            indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # calculate current beta for importance sampling
        beta = self.beta_start + (self.beta_end - self.beta_start) * min(1.0, self.frame / self.beta_frames)
        
        # get sampling probabilities
        if self.size > 0:
            probs = self.priorities[indices] ** self.alpha
            prob_sum = np.sum(self.priorities[:self.size].astype(np.float32) ** self.alpha)
            probs = probs / prob_sum if prob_sum > 0 else probs
        else:
            probs = np.ones_like(indices, dtype=np.float32) / len(indices)
        
        # calc importance-sampling weights=> (1/N * 1/P(i))^beta
        weights = (1.0 / self.size / probs) ** beta
        # normalize weights to have maximum weight equal to 1
        weights /= weights.max()
        # convert to tensor
        weights = torch.tensor(weights, dtype=torch.float, device=self.device)
        
        return indices, weights, probs

    # update priorities of sampled transitions
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            # ensure index is within bounds
            if 0 <= idx < self.size:
                self.priorities[idx] = priority

    # return current size of internal memory
    def __len__(self):
        return self.size
