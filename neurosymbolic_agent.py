# neurosymbolic_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

from dqn_agent_enhanced import DQNAgentEnhanced
from neural_symbolic_tree import NeuralGuidedDecisionTree

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeurosymbolicDQNAgent(DQNAgentEnhanced):
    """
    Neurosymbolic DQN Agent that extends the Enhanced DQN Agent
    with a Neural-Guided Decision Tree for symbolic reasoning.
    """
    
    def __init__(self, state_size, action_size, seed=0, 
                 learning_rate=0.0003, gamma=0.99, tau=0.0005,
                 buffer_size=100000, batch_size=128, update_every=8,
                 use_augmented_state=True, ksm_mode="off",
                 symbolic_guidance_weight=0.65):
        """Initialize the Neurosymbolic DQN Agent."""
        # Initialize the base Enhanced DQN Agent
        super().__init__(state_size, action_size, seed,
                        learning_rate, gamma, tau,
                        buffer_size, batch_size, update_every,
                        use_augmented_state, ksm_mode)
        
        # Create the neural-guided decision tree after the DQN is initialized
        self.symbolic_tree = NeuralGuidedDecisionTree(self.policy_net, device)
        
        # Weight for symbolic guidance (0 = pure neural, 1 = pure symbolic)
        # This is now a fixed value that won't change during training
        self.symbolic_guidance_weight = symbolic_guidance_weight
        
        # Track symbolic reasoning for analysis
        self.decision_history = []
    
    def act(self, state, eps=0.0):
        """
        Select an action using a hybrid of symbolic reasoning and epsilon-greedy policy.
        
        Args:
            state: Current environment state
            eps: Epsilon value for exploration
            
        Returns:
            int: Selected action
        """
        # Preprocess state
        state_vector = self.preprocess_state(state)
        state_tensor = torch.FloatTensor(state_vector).to(device)
        
        # List of possible actions
        possible_actions = list(range(self.action_size))
        
        # Get symbolic guidance
        guidance = self.symbolic_tree.guide_action_selection(state, state_tensor, possible_actions)
        
        # Store decision for analysis
        self.decision_history.append(guidance)
        
        # Check if we're exploring randomly
        if random.random() < eps:
            if guidance['mode'] == 'guided' and random.random() < 0.3:
                # 30% of exploration uses guided randomness based on the symbolic mask
                if guidance['action_mask'] is not None:
                    # Sample from action mask probabilities
                    probs = guidance['action_mask'] / guidance['action_mask'].sum()
                    return np.random.choice(possible_actions, p=probs)
            # 70% of exploration is pure random
            return random.choice(possible_actions)
        
        # If not exploring, use hybrid decision making
        with torch.no_grad():
            self.policy_net.eval()
            action_values = self.policy_net(state_tensor).cpu().data.numpy()
            self.policy_net.train()
            
            if guidance['mode'] == 'guided' and guidance['action_mask'] is not None:
                # Blend neural and symbolic preferences
                # Normalize neural values to [0,1] range for blending
                neural_prefs = action_values - action_values.min()
                if neural_prefs.max() > 0:
                    neural_prefs = neural_prefs / neural_prefs.max()
                
                # Normalize symbolic mask
                symbolic_prefs = guidance['action_mask']
                symbolic_prefs = symbolic_prefs / symbolic_prefs.max()
                
                # Weighted blend - using the fixed symbolic guidance weight
                w = self.symbolic_guidance_weight
                blended_prefs = (1 - w) * neural_prefs + w * symbolic_prefs
                
                # Select action with highest blended preference
                return np.argmax(blended_prefs)
            else:
                # Pure neural decision
                return np.argmax(action_values)
    
    def step(self, state, action, reward, next_state, done, info=None):
        """Process a step and learn if appropriate."""
        # Store symbolic decision
        symbolic_decision = self.decision_history[-1] if self.decision_history else None
        
        # Call the parent class step method for core functionality
        super().step(state, action, reward, next_state, done, info)
        
        # Reset symbolic decision history if episode ended
        if done:
            self.decision_history = []
            
    def get_symbolic_stats(self):
        """Get statistics about symbolic decision making."""
        if not hasattr(self, 'symbolic_stats'):
            self.symbolic_stats = {
                'neural_decisions': 0,
                'guided_decisions': 0,
                'key0_preference': 0,
                'key1_preference': 0,
                'door0_preference': 0,
                'door1_preference': 0
            }
            
        return self.symbolic_stats
    
    def update_symbolic_guidance_weight(self, success_rate, episode_num):
        """Return the fixed symbolic guidance weight - no adaptive adjustments."""
        # Simply return the fixed weight value - no changes
        return self.symbolic_guidance_weight
