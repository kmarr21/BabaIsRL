import numpy as np
import torch
import random
from collections import namedtuple

# Define transition tuple
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'done'))

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer for storing and sampling transitions."""
    
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=100000, device="cpu"):
        """Initialize a PrioritizedReplayBuffer.
        
        Args:
            capacity (int): Maximum size of buffer
            alpha (float): How much prioritization to use (0 = no prioritization, 1 = full prioritization)
            beta_start (float): Start value of importance-sampling correction (0 = no correction, 1 = full correction)
            beta_end (float): Final value of beta after annealing
            beta_frames (int): Number of frames over which to anneal beta
            device (str): Device to store tensors on
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.device = device
        
        # Current position in buffer
        self.position = 0
        # Current size of buffer (for before it's full)
        self.size = 0
        # Frame counter for beta annealing
        self.frame = 1
        
        # Buffers for experiences
        self.memory = []
        # Priorities with small epsilon to avoid zero probabilities
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
        # Track successful episodes separately - increased capacity
        self.success_memory = []
        self.success_count = 0
        self.max_success_episodes = 250  # Increased from 200 to 250
        
    def push(self, state, action, next_state, reward, done, is_success=False):
        """Store a transition in the buffer.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            reward: Reward received
            done: Whether episode terminated
            is_success: Whether this episode was successful (optional)
        """
        # Find max priority (use 1 for new experiences if buffer empty)
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        transition = Transition(state, action, next_state, reward, done)
        
        # For successful episodes, store separately
        if is_success and done:
            if self.success_count < self.max_success_episodes:
                self.success_memory.append(transition)
                self.success_count += 1
            elif self.success_count >= self.max_success_episodes:
                # If success buffer full, replace a random successful episode
                replace_idx = random.randint(0, self.max_success_episodes - 1)
                self.success_memory[replace_idx] = transition
        
        # Store in main memory
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
            self.size += 1
        else:
            self.memory[self.position] = transition
        
        # Update priority
        self.priorities[self.position] = max_priority
        
        # Update position
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, success_bias=0.4):
        """Sample a batch of transitions.
        
        Args:
            batch_size (int): Number of transitions to sample
            success_bias (float): Probability of sampling from successful episodes
        
        Returns:
            tuple: Batch of transitions, importance-sampling weights, and indices
        """
        # Increase frame counter (for beta annealing)
        self.frame += 1
        
        # Decide whether to sample from success memory
        use_success = (random.random() < success_bias) and self.success_count > 0
        
        if use_success:
            # Sample from successful episodes
            indices = np.random.choice(self.success_count, 
                                      min(batch_size, self.success_count), 
                                      replace=False)
            batch = [self.success_memory[idx] for idx in indices]
            # Use uniform weights for success samples
            weights = torch.ones(len(batch), device=self.device)
            
            # If we need more samples, get them from regular memory
            if len(batch) < batch_size and self.size > 0:
                regular_batch_size = batch_size - len(batch)
                regular_indices, regular_weights, _ = self._sample_proportional(regular_batch_size)
                
                batch.extend([self.memory[idx] for idx in regular_indices])
                # Combine weights
                weights = torch.cat([weights, regular_weights])
                indices = np.concatenate([indices, regular_indices])
            
        else:
            # Sample from regular memory based on priorities
            indices, weights, probs = self._sample_proportional(batch_size)
            batch = [self.memory[idx] for idx in indices]
        
        # Convert batch to tensor format
        states, actions, next_states, rewards, dones = zip(*batch)
        
        # Stack states and next_states if they're tensors
        if isinstance(states[0], torch.Tensor):
            states = torch.stack(states)
            next_states = torch.stack(next_states)
        
        # Convert other elements to tensors
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float, device=self.device)
        
        return (states, actions, next_states, rewards, dones), weights, indices
    
    def _sample_proportional(self, batch_size):
        """Sample indices based on priorities.
        
        Args:
            batch_size (int): Number of indices to sample
            
        Returns:
            tuple: Sampled indices, importance-sampling weights, and probabilities
        """
        if self.size < batch_size:
            # If buffer isn't full yet, sample from what we have
            indices = np.random.choice(self.size, self.size, replace=False)
            # Pad with random indices if needed
            if self.size < batch_size:
                indices = np.concatenate([indices, 
                                         np.random.choice(self.size, batch_size - self.size, replace=True)])
        else:
            # Sample from priorities
            priorities = self.priorities[:self.size]
            probs = priorities ** self.alpha
            probs /= probs.sum()
            
            indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Calculate current beta for importance sampling
        beta = self.beta_start + (self.beta_end - self.beta_start) * min(1.0, self.frame / self.beta_frames)
        
        # Get sampling probabilities
        if self.size > 0:
            probs = self.priorities[indices] ** self.alpha
            prob_sum = np.sum(self.priorities[:self.size].astype(np.float32) ** self.alpha)
            probs = probs / prob_sum if prob_sum > 0 else probs
        else:
            probs = np.ones_like(indices, dtype=np.float32) / len(indices)
        
        # Calculate importance-sampling weights: (1/N * 1/P(i))^beta
        weights = (1.0 / self.size / probs) ** beta
        # Normalize weights to have maximum weight equal to 1
        weights /= weights.max()
        # Convert to tensor
        weights = torch.tensor(weights, dtype=torch.float, device=self.device)
        
        return indices, weights, probs
    
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions.
        
        Args:
            indices (list): Indices to update
            priorities (list): New priorities
        """
        for idx, priority in zip(indices, priorities):
            # Ensure index is within bounds
            if 0 <= idx < self.size:
                self.priorities[idx] = priority
    
    def __len__(self):
        """Return the current size of internal memory."""
        return self.size
