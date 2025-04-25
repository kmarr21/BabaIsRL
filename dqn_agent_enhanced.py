import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple

from prioritized_replay_buffer import PrioritizedReplayBuffer, Transition

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    """Deep Q-Network with improved architecture for enhanced LIFO environment."""
    
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        
        # Larger network for more complex environment
        hidden_size1 = 256
        hidden_size2 = 256
        hidden_size3 = 128
        
        self.fc1 = nn.Linear(state_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, action_size)
        
        # Initialize weights with Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        
        # Add batch normalization for better training stability
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
    
    def forward(self, x):
        # Handle single sample (during action selection)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)

class DQNAgentEnhanced:
    """Agent implementing DQN with prioritized experience replay for enhanced LIFO environment."""
    
    def __init__(self, state_size, action_size, seed=0, 
                 learning_rate=0.0003, gamma=0.99, tau=0.001, 
                 buffer_size=100000, batch_size=64, update_every=4):
        """Initialize agent parameters."""
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Q-Networks (policy and target)
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in evaluation mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Prioritized Replay memory
        self.memory = PrioritizedReplayBuffer(buffer_size, device=device)
        
        # Initialize step counter
        self.t_step = 0
        
        # For tracking training progress
        self.loss_list = []
        
        # Success tracking
        self.current_episode_success = False
    
    def preprocess_state(self, state_dict):
        """Convert dictionary observation to flat vector with improved features for enhanced LIFO."""
        # Extract components
        agent_pos = state_dict['agent']
        enemies_pos = state_dict['enemies'].flatten()
        enemy_directions = state_dict['enemy_directions']
        keys_pos = state_dict['keys'].flatten()
        key_status = state_dict['key_status']
        doors_pos = state_dict['doors'].flatten()
        door_status = state_dict['door_status']
        key_stack = state_dict['key_stack']
        
        # Calculate distances to keys and doors using Manhattan distance
        distances_to_keys = []
        for i, key_pos in enumerate(state_dict['keys']):
            if key_status[i] == 0:  # Only include uncollected keys
                dist = np.abs(agent_pos[0] - key_pos[0]) + np.abs(agent_pos[1] - key_pos[1])
                distances_to_keys.append(dist)
            else:
                distances_to_keys.append(-1)  # -1 indicates collected key
        
        distances_to_doors = []
        for i, door_pos in enumerate(state_dict['doors']):
            if door_status[i] == 0:  # Only include unopened doors
                dist = np.abs(agent_pos[0] - door_pos[0]) + np.abs(agent_pos[1] - door_pos[1])
                distances_to_doors.append(dist)
            else:
                distances_to_doors.append(-1)  # -1 indicates opened door
        
        # Calculate distances to enemies
        distances_to_enemies = []
        for enemy_pos in state_dict['enemies']:
            dist = np.abs(agent_pos[0] - enemy_pos[0]) + np.abs(agent_pos[1] - enemy_pos[1])
            distances_to_enemies.append(dist)
        
        # Features related to key-door relationship
        keys_collected = np.sum(key_status)
        doors_opened = np.sum(door_status)
        keys_remaining = len(key_status) - keys_collected
        doors_remaining = len(door_status) - doors_opened
        
        # LIFO-specific features
        has_key = 1.0 if len(key_stack) > 0 and key_stack[0] >= 0 else 0.0
        
        # Next usable door features
        next_usable_door_dist = -1
        next_usable_door_idx = -1
        if len(key_stack) > 0 and key_stack[0] >= 0:  # If we have a key in the stack
            top_key = key_stack[0]
            if top_key < len(door_status) and door_status[top_key] == 0:  # If matching door exists and is not open
                door_pos = state_dict['doors'][top_key]
                next_usable_door_dist = np.abs(agent_pos[0] - door_pos[0]) + np.abs(agent_pos[1] - door_pos[1])
                next_usable_door_idx = top_key
        
        # One-hot encoding of top key in stack
        top_key_onehot = np.zeros(3)  # 2 keys + no key
        if len(key_stack) > 0 and key_stack[0] >= 0:
            top_key_onehot[key_stack[0]] = 1
        else:
            top_key_onehot[2] = 1  # No key
        
        # Combine all features into a single vector
        state_vector = np.concatenate([
            agent_pos,                  # Agent position (2)
            enemies_pos,                # Flattened enemy positions (2)
            enemy_directions,           # Enemy directions (1)
            key_status,                 # Key status (2)
            door_status,                # Door status (2)
            np.array(distances_to_keys, dtype=np.float32),      # Distances to keys (2)
            np.array(distances_to_doors, dtype=np.float32),     # Distances to doors (2)
            np.array(distances_to_enemies, dtype=np.float32),   # Distances to enemies (1)
            np.array([keys_collected, doors_opened, keys_remaining, doors_remaining], dtype=np.float32),  # Summary stats (4)
            np.array([has_key, next_usable_door_dist, next_usable_door_idx], dtype=np.float32),  # LIFO features (3)
            top_key_onehot,             # One-hot encoding of top key (3)
            key_stack                   # Full key stack (2)
        ])
        
        return state_vector
    
    def step(self, state, action, reward, next_state, done, info=None):
        """Process a step and learn if appropriate."""
        # Check if episode was successful
        if done and info is not None and 'success' in info:
            self.current_episode_success = info['success']
        elif done:
            self.current_episode_success = False
            
        # Preprocess states
        state_vector = self.preprocess_state(state)
        next_state_vector = self.preprocess_state(next_state)
        
        # Store experience in replay memory
        state_tensor = torch.FloatTensor(state_vector).to(device)
        next_state_tensor = torch.FloatTensor(next_state_vector).to(device)
        action_tensor = action
        reward_tensor = reward
        done_tensor = done
        
        self.memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor, done_tensor,
                        is_success=self.current_episode_success if done else False)
        
        # Learn every update_every steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            self.learn()
        
        # Reset episode success if episode ended
        if done:
            self.current_episode_success = False
    
    def act(self, state, eps=0.0):
        """Select an action using epsilon-greedy policy."""
        # Preprocess state
        state_vector = self.preprocess_state(state)
        state_tensor = torch.FloatTensor(state_vector).to(device)
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            with torch.no_grad():
                self.policy_net.eval()
                action_values = self.policy_net(state_tensor)
                self.policy_net.train()
                return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self):
        """Update value parameters using batch of experience tuples."""
        # Sample a batch from memory with priorities
        (state_batch, action_batch, next_state_batch, reward_batch, done_batch), importance_weights, indices = self.memory.sample(self.batch_size)
        
        # Convert to appropriate tensor shapes if needed
        if isinstance(action_batch, (list, tuple)):
            action_batch = torch.tensor(action_batch, dtype=torch.long, device=device)
        if isinstance(reward_batch, (list, tuple)):
            reward_batch = torch.tensor(reward_batch, dtype=torch.float, device=device)
        if isinstance(done_batch, (list, tuple)):
            done_batch = torch.tensor(done_batch, dtype=torch.float, device=device)
        
        # Compute Q values for current states
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute next state values (with target network) using Double DQN approach
        with torch.no_grad():
            # Use policy network to select actions
            next_q_values = self.policy_net(next_state_batch)
            next_actions = next_q_values.max(1)[1].unsqueeze(1)
            
            # Use target network to evaluate chosen actions
            next_state_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            next_state_values = next_state_values * (1 - done_batch)
        
        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # Calculate loss (with importance sampling weights for prioritized replay)
        td_errors = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduction='none')
        weighted_td_errors = importance_weights.unsqueeze(1) * td_errors
        loss = weighted_td_errors.mean()
        
        # Update priorities in replay buffer based on TD errors
        new_priorities = td_errors.detach().cpu().numpy() + 1e-6  # Small constant to avoid zero priorities
        self.memory.update_priorities(indices, new_priorities.reshape(-1))
        
        # Track loss
        self.loss_list.append(loss.item())
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()
        
        # Soft update target network
        self.soft_update()
    
    def soft_update(self):
        """Soft update target network parameters."""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, filename):
        """Save agent model and parameters."""
        checkpoint = {
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_list': self.loss_list
        }
        torch.save(checkpoint, filename)
    
    def load(self, filename):
        """Load agent model and parameters."""
        checkpoint = torch.load(filename, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_list = checkpoint['loss_list']
